import numpy as np
import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import os, os.path as osp
import importlib
import sys
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import copy
import torchmetrics
import shutil

import model_defs
import optimizers as optimizer_defs
import losses as loss_defs
import utils
from .helpers import *



class Base(object):

    def __init__(self, cfg, cfg_data, args, is_train, is_distributed, n_gpus):
        self.cfg = cfg
        self.cfg_data = cfg_data
        self.args = args
        self.is_train = is_train
        self.is_distributed = is_distributed
        self.n_gpus = n_gpus
        self.previous_teacher = None
        self.KD_replay = False
        # highest class index from past task
        self.last_valid_out_dim = 0 
        # highest class index from current task
        self.valid_out_dim = 0
        self.coreset_train = []
        self.ic = True  # full coreset or append the same number of samples as in self.file_list
        # synthetic data generation
        self.gen_inverted_samples = False


    def train(self, n_trial):
        print(f"Using GPU = {self.args.gpu} with (batch_size, workers) = ({self.cfg.batch_size}, {self.cfg.workers})")
        torch.cuda.set_device(self.args.gpu)

        self.cfg.num_total_classes = self.cfg_data.get_n_classes(self.args.split_type)
        # Load model
        self.model = model_defs.get_model(edict({'n_classes': self.cfg.num_total_classes,**self.cfg.model}))
        # Class mapping vars
        c = 0
        self.cfg.class_mapping = {}
        label_to_name = self.cfg_data.label_to_name[self.args.split_type]
        self.cfg.label_to_name_mapped = {}
        # Run tasks
        for current_t_index in range(self.cfg.increm.max_task):
            train_name = str(current_t_index)
            print('======================', train_name, '=======================')
            # Set variables depending on the task
            if current_t_index > 0:
                total_epochs_task = self.cfg.total_epochs_incremental_task
                self.cfg.total_epochs = self.cfg.total_epochs_incremental_task
                self.known_classes = self.valid_out_dim
                self.add_classes = self.cfg.increm.other_split_size
                self.valid_out_dim += self.cfg.increm.other_split_size
            else:
                total_epochs_task = self.cfg.total_epochs
                self.valid_out_dim = self.cfg.increm.first_split_size
                self.known_classes = 0
                self.add_classes = self.valid_out_dim

            # Load best checkpoint if desired. Otherwise, continue training from last checkpoint
            if current_t_index == 1 and self.cfg.increm.load_best_checkpoint_train:
                model_path = utils.get_best_model_path(osp.join(self.args.log_dir, f"task_{current_t_index - 1}"))
                assert model_path is not None, f"Model checkpoint not found in the log directory {self.args.log_dir}"
                print(f"=> loading checkpoint {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                epoch = checkpoint['epoch']
                utils.load_state_dict_single(checkpoint['state_dict'], self.model)
                print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}")
                del checkpoint

            model_defs.print_n_params(self.model)
            best_measure_info = utils.init_best_measure_info('acc', 'accuracy')
            log_dir_task = osp.join(self.args.log_dir, f"task_{train_name}")

            # load dataset for task
            self.train_dataset = getattr(importlib.import_module('.' + self.args.dataset, package='datasets'), 
                    'Dataset')('train', self.args.split_type, self.cfg_data, self.cfg.transforms['train'], 
                    self.add_classes, self.known_classes, rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial)

            self.val_dataset = getattr(importlib.import_module('.' + self.args.dataset, package='datasets'), 
                    'Dataset')('val', self.args.split_type, self.cfg_data, self.cfg.transforms['val'], 
                    self.add_classes, self.known_classes, rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial)

            print(f"Training classes: {self.train_dataset.keep_class_l}")
            # Class and label mapping
            for k in self.train_dataset.keep_class_l:
                self.cfg.class_mapping[str(k)] = c
                c += 1
            for prev_class, new_class in self.cfg.class_mapping.items():
                self.cfg.label_to_name_mapped[str(new_class)] = label_to_name[int(prev_class)]

            if current_t_index == 0 and self.cfg.increm.load_pretrained_task0:
                # Load pretrained model for task 0
                print("Loading pretrained model for task 0")
                self.valid_out_dim = self.cfg.increm.first_split_size
                # Create log dir if it does not exist
                if not osp.exists(osp.join(log_dir_task, 'checkpoints')):
                    os.makedirs(osp.join(log_dir_task, 'checkpoints'))
                # Copy checkpoint to log dir
                pretrained_checkpoint_path = osp.join('/ogr_cmu/models', self.args.dataset, self.cfg.model.name, f"trial_{n_trial+1}", 'checkpoints', 'model_best.pth.tar')
                assert osp.exists(pretrained_checkpoint_path), f"Pretrained checkpoint not found in {pretrained_checkpoint_path}"
                shutil.copy(pretrained_checkpoint_path, osp.join(log_dir_task, 'checkpoints', 'model_best.pth.tar'))

                model_path = utils.get_best_model_path(osp.join(self.args.log_dir, f"task_0"))
                assert model_path is not None, f"Model checkpoint not found in the log directory {self.args.log_dir}"
                print(f"=> loading checkpoint {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                epoch = checkpoint['epoch']
                utils.load_state_dict_single(checkpoint['state_dict'], self.model)
                self.model.cuda(self.args.gpu)
                
            else:
                self.train_sampler = None
                self.val_sampler = None

                # Append coreset samples to the train/val datasets if memory > 0
                self.train_dataset.append_coreset(self.coreset_train, self.ic, only=False)

                g = torch.Generator()
                g.manual_seed(3407)

                self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=(self.train_sampler is None),
                    num_workers=self.cfg.workers, pin_memory=True, sampler=self.train_sampler, drop_last=True if self.n_gpus>1 else False,
                    worker_init_fn=utils.seed_worker, generator=g)

                self.val_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=(self.val_sampler is None),
                    num_workers=self.cfg.workers, pin_memory=True, sampler=self.val_sampler, drop_last=True if self.n_gpus>1 else False,
                    worker_init_fn=utils.seed_worker, generator=g)   

                # Generate inverted samples
                if self.gen_inverted_samples:
                    self.generate_inverted_samples(self.previous_teacher, self.cfg.increm.learner.n_samples_per_class, self.cfg.increm.learner.inversion_batch_size, self.cfg.batch_size, log_dir_task, self.args.log_dir)

                # Modify the base LR if current_task_index > 0
                if current_t_index > 0:
                    self.cfg.optimizer.lr = self.cfg.optimizer.lr_incremental_task
                    if not self.cfg.optimizer.include_scheduler and 'scheduler' in self.cfg.optimizer :
                        del self.cfg.optimizer.scheduler
                        del self.cfg.optimizer['scheduler']
                        self.cfg.step_per_epoch = False
                        self.cfg.step_per_batch = False
                        print("Scheduler deleted")
                self.optimizer, self.scheduler = optimizer_defs.get_optimizer_scheduler(self.model, edict({**self.cfg.optimizer, 
                                        'total_epochs': total_epochs_task, 'n_steps_per_epoch': len(self.train_loader)}) )

                self.criteria = loss_defs.get_losses(self.cfg.loss, self.valid_out_dim)
                
                resume_checkpoint_path = utils.get_last_checkpoint_path(log_dir_task)
                if resume_checkpoint_path :
                    print(f"=> loading checkpoint {resume_checkpoint_path}")
                    checkpoint = torch.load(resume_checkpoint_path, map_location=torch.device('cpu'))
                    start_epoch = checkpoint['epoch'] + 1
                    if start_epoch >= total_epochs_task :
                        print(f"Start epoch {start_epoch} is greater than total epochs {total_epochs_task}")
                        sys.exit()
                    utils.load_state_dict_single(checkpoint['state_dict'], self.model, self.optimizer, self.scheduler, )
                    print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}")
                    del checkpoint

                else :
                    start_epoch = 1
                    print("=> no checkpoint found for resuming.")

                # Freeze backbone if desired (from the second task onwards)
                if current_t_index > 0 and self.cfg.increm.freeze_feature_extractor:
                    print("Freezing feature extractor...")
                    self.freeze_model(feature_extractor=True)

                # Freeze the weights for the previous classes in the classification layer if desired (from the second task onwards)
                if current_t_index > 0 and self.cfg.increm.freeze_classifier:
                    # Copy the weights and biases of the final linear layer
                    self.prev_weights = torch.empty_like(self.model.final.weight.data).copy_(self.model.final.weight.data)
                    self.prev_bias = torch.empty_like(self.model.final.bias.data).copy_(self.model.final.bias.data)

                # transfer models
                self.model.cuda(self.args.gpu)

                # transfer optimizers and schedulers
                optimizer_defs.optimizer_to_cuda(self.optimizer, self.args.gpu)
                optimizer_defs.scheduler_to_cuda(self.scheduler, self.args.gpu)

                if self.args.gpu == 0 :
                    train_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'train'))
                    val_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'val'))

                    # epoch, train, val bars
                    print('Printing progress info for GPU 0 only ...')
                    ebar = tqdm(total=total_epochs_task - start_epoch + 1, leave=True, desc='epoch', dynamic_ncols=False)
                    tbar = tqdm(total=len(self.train_loader), leave=True, desc='train', dynamic_ncols=False)
                    vbar = tqdm(total=len(self.val_loader), leave=True, desc='val', dynamic_ncols=False)

                step_per_epoch = False
                if 'scheduler' in self.cfg.optimizer :
                    if 'step_per_epoch' in self.cfg.optimizer.scheduler :
                        step_per_epoch = self.cfg.optimizer.scheduler.step_per_epoch

                for epoch in range(start_epoch, total_epochs_task + 1) :
                    torch.cuda.empty_cache()

                    self.train_epoch(tbar if self.args.gpu==0 else None, epoch, train_logger if self.args.gpu==0 else None, current_t_index)

                    measures = self.validate_epoch(vbar if self.args.gpu==0 else None, epoch, val_logger if self.args.gpu==0 else None) 

                    if self.args.gpu == 0 :
                        is_best = best_measure_info.func(measures[best_measure_info.tag],best_measure_info.val)
                        if is_best :
                            best_measure_info.val = measures[best_measure_info.tag]            

                        train_logger.flush()
                        val_logger.flush()           

                        if (epoch % self.args.save_epoch_freq == 0) and (self.args.gpu == 0) :
                            # save model 
                            state_dict = utils.get_state_dict_single(self.model, self.optimizer, self.scheduler, self.is_distributed)

                            utils.save_checkpoint(log_dir_task,
                                { 
                                    'epoch': epoch, 
                                    'state_dict': state_dict, 
                                    'best_measure_tag': best_measure_info.tag,
                                    'best_measure': best_measure_info.val, 
                                },
                                epoch,
                                save_last_only=self.args.save_last_only,
                                is_best=is_best,
                            )    

                    if step_per_epoch :
                        optimizer_defs.step_scheduler(self.scheduler)

                    if self.args.gpu == 0 :
                        ebar.update()
                        ebar.set_postfix(dict(epoch=epoch)) 

            self.KD_replay = True
            self.last_valid_out_dim = self.valid_out_dim

            # set to eval mode
            self.model.eval()
            # new teacher
            if self.cfg.increm.learner.type == 'deep_inversion' or self.cfg.increm.learner.type == 'abd':
                self.sample_shape = (-1, self.cfg.seq_len, self.cfg.model.n_joints, self.cfg.in_channels)
                self.previous_teacher = Teacher_v2(solver=copy.deepcopy(self.model), 
                                                    sample_shape = self.sample_shape, 
                                                    iters = self.power_iters, 
                                                    deep_inv_params = self.deep_inv_params, 
                                                    class_idx = np.arange(self.valid_out_dim), 
                                                    num_inverted_class = self.add_classes,
                                                    num_known_classes = self.known_classes,
                                                    config = self.cfg)
                self.previous_linear = copy.deepcopy(self.model.final)
                self.gen_inverted_samples = True
                self.inversion_replay = True
            else:
                self.previous_teacher = Teacher_v1(solver=copy.deepcopy(self.model))

            # Update coreset for train/val datasets if memory > 0
            if self.cfg.increm.memory > 0:
                self.coreset_train = self.train_dataset.update_coreset(self.coreset_train, self.cfg.increm.memory, 
                    np.arange(self.last_valid_out_dim), self.cfg.class_mapping)


            if self.args.gpu == 0 and not (current_t_index == 0 and self.cfg.increm.load_pretrained_task0):
                ebar.close()
                tbar.close()
                vbar.close()
                train_logger.close()
                val_logger.close()

        # save config 
        if self.args.gpu == 0 :
            # Save config edict object 
            utils.stdio.save_pickle(osp.join(self.args.log_dir, 'config.pkl'), self.cfg)

    
    def train_epoch(self, tbar, epoch, train_logger, current_t_index) :

        losses = edict({
            name: utils.AverageMeter() for name in self.criteria
        })
        
        # Class to save epoch metrics 
        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)
        n_batches = len(self.train_loader)

        # set to train mode
        self.model.train()

        # set epochs
        if self.train_sampler is not None :
            self.train_sampler.set_epoch(self.train_sampler.epoch + 1)

        if self.args.gpu == 0 :
            tbar.reset(total=n_batches)
            tbar.refresh()

        step_per_batch = False
        if 'scheduler' in self.cfg.optimizer :
            if 'step_per_batch' in self.cfg.optimizer.scheduler :
                step_per_batch = self.cfg.optimizer.scheduler.step_per_batch

        iter_loader = iter(self.train_loader)
        bi = 1
        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
        
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            output = self.model(pts)[:, :self.valid_out_dim]

            loss_tensors = []

            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                lval = lfunc(output, target)
                losses[lname].update(lval.item(), output.size(0) )
                loss_tensors.append(lweight * lval)

            loss = sum(loss_tensors)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if current_t_index > 0 and self.cfg.increm.freeze_classifier:
                # Restore the weights and biases for previous classes
                self.model.final.weight.data[:self.last_valid_out_dim] = self.prev_weights[:self.last_valid_out_dim]
                self.model.final.bias.data[:self.last_valid_out_dim] = self.prev_bias[:self.last_valid_out_dim]

            train_acc= acc_meter(output, target) * 100

            if step_per_batch :
                optimizer_defs.step_scheduler(self.scheduler)

            if self.args.gpu == 0 :           
                tbar.update()
                tbar.set_postfix({
                        'it': bi,
                        'loss': loss.item(), 
                        'train_acc': train_acc.item(),
                })
                tbar.refresh()

            bi += 1

        if self.args.gpu == 0 :
            acc_all = acc_meter.compute() * 100
            # hyperparam update
            train_logger.update(
                {'learning_rate': self.optimizer.param_groups[0]['lr']},
                step=epoch, prefix="stepwise")

            # loss update
            train_logger.update(
                { ltype: lmeter.avg for ltype, lmeter in losses.items() },
                step=epoch, prefix="loss")

            # measures update
            train_logger.update({
                'mean': acc_all,
                }, step=epoch, prefix="acc" )                

            acc_meter.reset()
            train_logger.flush()  
       

    @torch.no_grad()
    def validate_epoch(self, vbar, epoch, val_logger) :
   
        losses = edict({
            name: utils.AverageMeter() for name in self.criteria
        })
        
        # Class to save epoch metrics
        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)

        # set to eval mode
        self.model.eval()

        if self.args.gpu == 0 :
            vbar.reset(total=len(self.val_loader))
            vbar.refresh()

        n_batches = len(self.val_loader)
        iter_loader = iter(self.val_loader)
        bi = 1

        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            output = self.model(pts)[:, :self.valid_out_dim]

            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                lval = lfunc(output, target)
                losses[lname].update(lval.item(), output.size(0))
                loss_tensors.append(lweight * lval)

            loss = sum(loss_tensors)

            val_acc = acc_meter(output, target) * 100

            if self.args.gpu == 0 :
                vbar.update()
                vbar.set_postfix({
                    'it': bi,
                    'loss': loss.item(), 
                    'val_acc': val_acc.item(),
                })                
                vbar.refresh()

            bi += 1   

        if self.args.gpu == 0 :
            acc_all = acc_meter.compute() * 100
            
            # loss update
            val_logger.update(
                { ltype: lmeter.avg for ltype, lmeter in losses.items() },
                step=epoch, prefix="loss")

            # measures update
            val_logger.update({
                'mean': acc_all,
                }, step=epoch, prefix="acc" )                

            acc_meter.reset()               
            val_logger.flush() 

            return_values = { ltype: lmeter.avg for ltype, lmeter in losses.items() }
            return_values['acc'] = acc_all
            
            return return_values


    def evaluate(self, n_trial):
        is_testval = (self.args.train==0)
        mode = 'testval' if is_testval else 'test'

        # load config
        cfg = utils.stdio.load_pickle(osp.join(self.args.log_dir, 'config.pkl'))

        # Test each task
        for current_t_index in range(cfg.increm.max_task):
            if current_t_index > 0:
                self.valid_out_dim += self.cfg.increm.other_split_size
            else:
                self.valid_out_dim = self.cfg.increm.first_split_size
            # print name
            cfg.test_name = str(current_t_index)
            log_dir_task = osp.join(self.args.log_dir, f"task_{cfg.test_name}")
            print('======================', cfg.test_name, '=======================')
            # Load model
            self.model = model_defs.get_model(edict({'n_classes': cfg.num_total_classes,**cfg.model}) )
            model_defs.print_n_params(self.model)
            for test_mode in ['local', 'global', 'old', 'new']:
                if current_t_index > 0:
                    if test_mode == 'local':
                        self.known_classes = self.valid_out_dim - self.cfg.increm.other_split_size
                        self.add_classes = self.cfg.increm.other_split_size
                    elif test_mode == 'global':
                        self.known_classes = 0
                        self.add_classes = self.valid_out_dim 
                    elif test_mode == 'old':
                        self.known_classes = 0
                        self.add_classes = self.cfg.increm.first_split_size
                    elif test_mode == 'new':
                        self.known_classes = self.cfg.increm.first_split_size
                        self.add_classes = self.valid_out_dim - self.cfg.increm.first_split_size
                else:
                    self.known_classes = 0
                    self.add_classes = self.valid_out_dim
                cfg.test_mode = test_mode
                print('======================', cfg.test_mode, '=======================')
                # define dataset
                self.test_dataset = getattr(importlib.import_module('.' + self.args.dataset, package='datasets'), 
                    'Dataset')(mode, self.args.split_type, self.cfg_data, self.cfg.transforms[mode], 
                    self.add_classes, self.known_classes, rm_global_scale=self.cfg.rm_global_scale, drop_seed=n_trial)       

                g = torch.Generator()
                g.manual_seed(3407)

                self.test_loader = DataLoader(self.test_dataset, batch_size=cfg.batch_size, shuffle=False,
                    num_workers=cfg.workers, pin_memory=True, sampler=None, drop_last=False,
                    worker_init_fn=utils.seed_worker, generator=g)
                
                print(f"Testing classes: {self.test_dataset.keep_class_l}")
                # load checkpoint
                cfg.increm.load_best_checkpoint_test = self.cfg.increm.load_best_checkpoint_test
                if self.cfg.increm.load_best_checkpoint_test and current_t_index == 0:
                    model_path = utils.get_best_model_path(log_dir_task)
                else:
                    model_path = utils.get_last_checkpoint_path(log_dir_task)
                assert model_path is not None, \
                    f"Model checkpoint not found in the log directory {log_dir_task}"
                print(f"=> loading checkpoint {model_path}")
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                utils.load_state_dict_single(checkpoint['state_dict'], self.model )
                print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}")
                del checkpoint

                # transfer models
                self.model.cuda(self.args.gpu)

                # evaluate
                self.cfg = cfg
                self.evaluate_task()
            
        # Save results for all tasks
        if cfg.increm.max_task > 1:
            self.save_accuracies_task()
            
    
    @torch.no_grad()
    def evaluate_task(self):
        # Class to save epoch metrics
        acc_meter = utils.Meter(self.valid_out_dim, self.cfg.label_to_name_mapped) 
        acc_meter_torchmetrics = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)

        # set to eval mode
        self.model.eval()

        tbar = tqdm(total=len(self.test_loader), leave=True, desc='test', dynamic_ncols=False)
        tbar.refresh()

        n_batches = len(self.test_loader)

        iter_loader = iter(self.test_loader)
        bi = 1

        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            output = self.model(pts)[:, :self.valid_out_dim]

            acc_meter.update(output, target)
            test_acc_= acc_meter_torchmetrics(output, target) * 100

            tbar.update()   
            tbar.set_postfix({
                    'it': bi,
                    'test_acc': test_acc_.item(),
                })           
            tbar.refresh()

            bi += 1

        tbar.close()

        test_folder = osp.join(self.args.log_dir, f"task_{self.cfg.test_name}",'test')
        if not osp.exists(test_folder):
            os.makedirs(test_folder)

        if self.args.save_conf_mat :
            conf_mat = acc_meter.conf_matrix.squeeze().cpu().numpy()

            utils.save_conf_mat_image(
                conf_mat, 
                self.cfg.label_to_name_mapped,
                osp.join(test_folder, f"conf_mat_{self.cfg.test_mode}.png"),
            )

        acc_all = acc_meter.accuracies()
        acc_all_torchmetrics = acc_meter_torchmetrics.compute() * 100
        acc_meter_torchmetrics.reset()
        with open(osp.join(test_folder, f"test_metrics_{self.cfg.test_mode}.json"), 'w') as f:
            json.dump({'Acc': acc_all, 'Acc_torchmetrics': acc_all_torchmetrics.item()}, f, indent=4)


    def save_accuracies_task(self):
        # Save results for all tasks
        metrics_dict = {}
        metrics_dict['local'] = []
        metrics_dict['global'] = []
        metrics_dict['old'] = []
        metrics_dict['new'] = []
        for task in range(self.cfg.increm.max_task):
            test_folder = osp.join(self.args.log_dir, f"task_{task}",'test')
            for test_mode in ['local', 'global', 'old', 'new']:
                with open(osp.join(test_folder, f"test_metrics_{test_mode}.json"), 'r') as f:
                    metrics = json.load(f)
                metrics_dict[test_mode].append(metrics['Acc_torchmetrics'])
        
        # Save results in json files
        for test_mode in ['local', 'global', 'old', 'new']:
            with open(osp.join(self.args.log_dir, f"test_metrics_{test_mode}.json"), 'w') as f:
                json.dump({'acc_metrics': metrics_dict[test_mode]}, f, indent=4)

        # Save results in a single matplotlib figure
        color_l = [(0.5, 0.5, 0.9), (0.9, 0.5, 0.5), (0.5, 0.9, 0.5), (0.9, 0.9, 0.5)]
        fig, ax = plt.subplots()
        x_index = range(1, self.cfg.increm.max_task+1)

        ax.plot(x_index, metrics_dict['local'], '-o', color=color_l[0])
        ax.plot(x_index, metrics_dict['global'], '-o', color=color_l[1])
        ax.plot(x_index, metrics_dict['old'], '-o', color=color_l[2])
        ax.plot(x_index, metrics_dict['new'], '-o', color=color_l[3])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('% Accuracy', fontsize=14)
        ax.set_xlabel('Task', fontsize=14)
        plt.xticks(x_index)
        ax.set_title('Accuracies per task', fontsize=16)
        ax.legend(['local', 'global', 'old', 'new'])
        plt.ylim([0, 105])

        fig.tight_layout()
        fig.savefig(osp.join(self.args.log_dir, 'accuracies_per_task.png'))
        plt.close(fig)

    
    def freeze_model(self, feature_extractor=False, classifier=False):
        if feature_extractor:
            # Freeze initial layer
            for param in self.model.initial.parameters():
                param.requires_grad = False
            # Freeze spatial_att
            for param in self.model.spatial_att.parameters():
                param.requires_grad = False
            # Freeze temporal_att 
            for param in self.model.temporal_att.parameters():
                param.requires_grad = False
        if classifier:
            # Freeze classifier
            for param in self.model.final.parameters():
                param.requires_grad = False



    

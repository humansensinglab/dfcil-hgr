import torch
from easydict import EasyDict as edict
import torchmetrics
import numpy as np  
import os, os.path as osp
import importlib
import sys
from tqdm import tqdm
import copy
import torchmetrics
import shutil

import optimizers as optimizer_defs
import utils
import model_defs
from .base import Base
import losses as loss_defs
from .helpers import *


class Rdfcil(Base):

    def __init__(self, cfg, cfg_data, args, is_train, is_distributed, n_gpus):
        super(Rdfcil, self).__init__(cfg, cfg_data, args, is_train, is_distributed, n_gpus)
        self.inversion_replay = False
        self.KD_replay = False
        self.power_iters = self.cfg.increm.learner.power_iters
        self.deep_inv_params = self.cfg.increm.learner.deep_inv_params
        self.lambda_ce =self.cfg.increm.learner.lambda_loss[0]
        self.lambda_hkd =self.cfg.increm.learner.lambda_loss[1]
        self.lambda_rkd =self.cfg.increm.learner.lambda_loss[2]
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
            # print name
            train_name = str(current_t_index)
            print('======================', train_name, '=======================')
            # Set variables depending on the task
            if current_t_index > 0:
                total_epochs_task = self.cfg.total_epochs_incremental_task
                self.cfg.total_epochs = self.cfg.total_epochs_incremental_task
                self.known_classes = self.valid_out_dim
                self.add_classes = self.cfg.increm.other_split_size
                self.valid_out_dim += self.cfg.increm.other_split_size

                self.alpha = math.log(self.valid_out_dim / 2 + 1, 2)
                beta2 = self.known_classes / self.valid_out_dim
                self.beta = math.sqrt(beta2)
                self.cls_count = torch.zeros(self.valid_out_dim)
                self.cls_weight = torch.ones(self.valid_out_dim)

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

                # Append coreset samples to the train/val datasets
                self.train_dataset.append_coreset(self.coreset_train, self.ic, only=False)
                
                g = torch.Generator()
                g.manual_seed(3407)
                
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=(self.train_sampler is None),
                    num_workers=self.cfg.workers, pin_memory=True, sampler=self.train_sampler, drop_last=True if self.n_gpus>1 else False,
                    worker_init_fn=utils.seed_worker, generator=g)

                self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, shuffle=(self.val_sampler is None),
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

                self.criteria = loss_defs.get_losses(self.cfg.loss, self.valid_out_dim)
                if 'rkd' in self.criteria:
                    module = nn.ModuleList([self.model, self.criteria['rkd'].func])
                    self.criteria['rkd'].func.cuda(self.args.gpu)
                    self.optimizer, self.scheduler = optimizer_defs.get_optimizer_scheduler(module, edict({**self.cfg.optimizer, 
                                        'total_epochs': total_epochs_task, 'n_steps_per_epoch': len(self.train_loader)}) )
                else:
                    self.optimizer, self.scheduler = optimizer_defs.get_optimizer_scheduler(self.model, edict({**self.cfg.optimizer, 
                                        'total_epochs': total_epochs_task, 'n_steps_per_epoch': len(self.train_loader)}) )

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
                    
                    if epoch > self.cfg.increm.learner.start_finetuning_epochs:
                        self.optimizer.param_groups[0]['lr'] = self.cfg.optimizer.lr_finetune_incremental_task

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
            if self.cfg.increm.learner.type in ['deep_inversion', 'abd','rdfcil']:
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
        acc_meter_global = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)

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

            # if synthetic data replay
            if self.inversion_replay:
                pts_replay, target_replay, target_replay_hat = self.sample(self.previous_teacher)
                # Send to GPU
                pts_replay = pts_replay.cuda(self.args.gpu)
                target_replay = target_replay.cuda(self.args.gpu)
                target_replay_hat = target_replay_hat.cuda(self.args.gpu)

            # combine inputs and generated samples for classification
            if self.inversion_replay:
                pts_com, target_com = self.combine_data(((pts, target),(pts_replay, target_replay)))
            else:
                pts_com, target_com = pts, target

            # forward pass
            features = self.model.forward_feature(pts_com)
            output = self.model.final(features)[:, :self.valid_out_dim]

            batch_idx = np.arange(len(pts))
            kd_index = np.arange(len(pts), len(pts_com))
            loss_tensors = []

            indices, counts = target_com.cpu().unique(return_counts=True)
            self.cls_count[indices] += counts
            if self.inversion_replay:
                cls_weight = self.cls_count.sum() / self.cls_count.clamp(min=1)
                self.cls_weight = cls_weight.div(cls_weight.min())

            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight

                if self.inversion_replay:
                    if lname == 'hkd':
                        with torch.no_grad(): 
                            prev_logits = self.previous_linear(self.previous_teacher.generate_scores_pen(pts_com[kd_index, :])).detach()
                        lval = self.lambda_hkd * self.alpha * self.beta * lfunc(output[kd_index,:self.last_valid_out_dim], prev_logits[:,:self.last_valid_out_dim]) 
                        losses[lname].update(lval, len(kd_index) )
                        loss_tensors.append(lweight * lval)
                    
                    elif lname == 'rkd' :
                        if epoch <= self.cfg.increm.learner.start_finetuning_epochs:
                            input_rkd = features[batch_idx,:]
                            target_rkd = self.previous_teacher.generate_scores_pen(pts_com[batch_idx,:])
                            lval =self.lambda_rkd * self.alpha * self.beta * lfunc(input_rkd, target_rkd) 
                            losses[lname].update(lval, len(batch_idx) )
                            loss_tensors.append(lweight * lval)    

                    elif lname == 'rdfcil_local_cross_entropy' :
                        if epoch <= self.cfg.increm.learner.start_finetuning_epochs:
                            #lce
                            if self.cfg.increm.other_split_size == 1:
                                lval= (self.lambda_ce * (1 + 1 / self.alpha) / self.beta) * lfunc(output[batch_idx,self.last_valid_out_dim:self.valid_out_dim], torch.unsqueeze((target[batch_idx]-self.last_valid_out_dim + 1).float(), dim=1), binary=True)
                            else:
                                lval= (self.lambda_ce * (1 + 1 / self.alpha) / self.beta) * lfunc(output[batch_idx,self.last_valid_out_dim:self.valid_out_dim], (target[batch_idx]-self.last_valid_out_dim).long())
                            losses[lname].update(lval, len(batch_idx) )
                        else:
                            #gce 
                            with torch.no_grad():             
                                feature = self.model.forward_feature(pts_com).detach()
                            ft_weight = self.cls_weight.cuda()
                            lval= self.lambda_ce * lfunc(self.model.final(feature)[:, :self.valid_out_dim], target_com, weight=ft_weight)
                            losses[lname].update(lval, len(pts_com) )
                        loss_tensors.append(lweight * lval)
                else:
                    lval = lfunc(output[batch_idx], target_com[batch_idx])
                    losses[lname].update(lval, len(batch_idx) )
                    loss_tensors.append(lweight * lval)

            loss = sum(loss_tensors)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            if  self.cfg.increm.freeze_classifier and current_t_index > 0:
                # Restore the weights and biases for previous classes
                self.model.final.weight.data[:self.last_valid_out_dim] = self.prev_weights[:self.last_valid_out_dim]
                self.model.final.bias.data[:self.last_valid_out_dim] = self.prev_bias[:self.last_valid_out_dim]

            train_acc_global = acc_meter_global(output, target_com) * 100

            if step_per_batch:
                optimizer_defs.step_scheduler(self.scheduler)


            if self.args.gpu == 0 :           
                tbar.update()
                tbar.set_postfix({
                        'it': bi,
                        'loss': loss.item(), 
                        'train_acc_global': train_acc_global.item(),
                })
                tbar.refresh()

            bi += 1

        if self.args.gpu == 0 :
            acc_all = acc_meter_global.compute() * 100

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
                'global': acc_all,
                }, step=epoch, prefix="acc" )  

            acc_meter_global.reset()

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
                if lname in ['rkd','hkd', 'rdfcil_local_cross_entropy']:
                    lval = torch.zeros((1,), requires_grad=True).cuda()
                else:
                    lval = lfunc(output, target)
                losses[lname].update(lval.item(), output.size(0) )
                loss_tensors.append(lweight * lval)

            loss = sum(loss_tensors)
            
            val_acc= acc_meter(output, target) * 100

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
                'global': acc_all,
                }, step=epoch, prefix="acc" ) 
                                 
            acc_meter.reset()   
            val_logger.flush() 

            return_values = { ltype: lmeter.avg for ltype, lmeter in losses.items() }
            return_values['acc'] = acc_all
            
            return return_values


    def sample(self, teacher):
        return teacher.sample()

    def generate_inverted_samples(self, teacher, size, inversion_batch_size, dataloader_batch_size, log_dir,inverted_sample_dir):
        return teacher.generate_inverted_samples(size, inversion_batch_size, dataloader_batch_size, log_dir,inverted_sample_dir)

    def combine_data(self, data):
        x, y = [],[]
        for i in range(len(data)):
            x.append(data[i][0])
            y.append(data[i][1])
        x, y = torch.cat(x), torch.cat(y)
        return x, y


    

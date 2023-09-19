import torch
from torch.nn import functional as F
from easydict import EasyDict as edict
import torchmetrics
import numpy as np  

import optimizers as optimizer_defs
import utils
from .base import Base


class DeepInversion(Base):

    def __init__(self, cfg, cfg_data, args, is_train, is_distributed, n_gpus):
        super(DeepInversion, self).__init__(cfg, cfg_data, args, is_train, is_distributed, n_gpus)
        self.inversion_replay = False
        self.KD_replay = False
        self.power_iters = self.cfg.increm.learner.power_iters
        self.deep_inv_params = self.cfg.increm.learner.deep_inv_params
        self.gen_inverted_samples = False

    
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

            # if KD -> generate pseudo labels
            if self.KD_replay:
                allowed_predictions = np.arange(self.last_valid_out_dim)
                target_hat = self.previous_teacher.generate_scores(pts, allowed_predictions=allowed_predictions)
                _, target_hat_com = self.combine_data(((pts, target_hat),(pts_replay, target_replay_hat[:,:self.last_valid_out_dim])))
            else:
                target_hat_com = None

            # combine inputs and generated samples for classification
            if self.inversion_replay:
                pts_com, target_com = self.combine_data(((pts, target),(pts_replay, target_replay)))
            else:
                pts_com, target_com = pts, target

            # forward pass
            output = self.model(pts_com)[:, :self.valid_out_dim]

            #batch_idx = np.arange(self.cfg.batch_size)
            batch_idx = np.arange(len(pts))
            kd_index = np.arange(len(pts), len(pts_com))
            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                if lname == 'distillation' :
                    # KD old
                    if  target_hat_com is not None:
                        lval = lfunc(output[batch_idx], target_hat_com[batch_idx], allowed_predictions=np.arange(self.last_valid_out_dim).tolist())
                        losses[lname].update(lval.item(), output.size(0) )
                    else:
                        lval = 0.
                        losses[lname].update(lval, output.size(0) )
                    loss_tensors.append(lweight * lval)
                    # KD new
                    if  target_hat_com is not None and len(kd_index) > 0:
                        target_hat_com = F.softmax(target_hat_com[:, :self.last_valid_out_dim] / self.cfg.loss.distillation.temperature, dim=1)
                        target_hat_com = [target_hat_com]
                        target_hat_com.append(torch.zeros((len(target_com),self.valid_out_dim-self.last_valid_out_dim), requires_grad=True).cuda())
                        target_hat_com = torch.cat(target_hat_com, dim=1)
                        lval = lfunc(output[kd_index], target_hat_com[kd_index], allowed_predictions=np.arange(self.valid_out_dim).tolist(), T=self.cfg.loss.distillation.temperature, soft_t=True)
                        losses[lname].update(lval.item(), output.size(0) )
                    else:
                        lval = 0.
                        losses[lname].update(lval, output.size(0) )
                    loss_tensors.append(lweight * lval)
                else:
                    lval = lfunc(output[batch_idx], target_com[batch_idx])
                    losses[lname].update(lval, output.size(0) )
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

            if step_per_batch :
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

            # if KD -> generate pseudo labels
            if self.KD_replay:
                allowed_predictions = list(range(self.last_valid_out_dim))
                output_hat = self.previous_teacher.generate_scores(pts, allowed_predictions=allowed_predictions)
            else:
                output_hat = None

            output = self.model(pts)[:, :self.valid_out_dim]

            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                if lname == 'distillation' :
                    if  output_hat is not None:
                        lval = lfunc(output, output_hat, allowed_predictions=allowed_predictions)
                        losses[lname].update(lval.item(), output.size(0) )
                    else:
                        losses[lname].update(0., output.size(0) )
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


    

import torch
from easydict import EasyDict as edict
import torchmetrics
import numpy as np  

import optimizers as optimizer_defs
import utils
import model_defs
from .base import Base


class AlwaysBeDreaming(Base):

    def __init__(self, cfg, cfg_data, args, is_train, is_distributed, n_gpus):
        super(AlwaysBeDreaming, self).__init__(cfg, cfg_data, args, is_train, is_distributed, n_gpus)
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
            features = self.model.forward_feature(pts_com)
            output = self.model.final(features)[:, :self.valid_out_dim]

            batch_idx = np.arange(len(pts))
            kd_index = np.arange(len(pts_com))
            loss_tensors = []

            # class balancing
            mappings = torch.ones(target_com.size(), dtype=torch.float32)
            mappings = mappings.cuda()
            rnt = 1.0 * self.last_valid_out_dim / self.valid_out_dim
            mappings[:len(pts)] = 1-rnt
            mappings[len(pts):] = rnt
            dw_cls = mappings
            dw_KD = 1

            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                if lname == 'abd_distillation' :
                    if  target_hat_com is not None:
                       # hard - linear
                        logits_KD = self.previous_linear(features[kd_index])[:,:self.last_valid_out_dim]
                        logits_KD_past = self.previous_linear(self.previous_teacher.generate_scores_pen(pts_com[kd_index]))[:,:self.last_valid_out_dim]
                        lval = lfunc(logits_KD, logits_KD_past,dw_KD)
                    else:
                        lval = torch.zeros((1,), requires_grad=True).cuda()
                    losses[lname].update(lval.item(), output.size(0) )
                    loss_tensors.append(lweight * lval)
                elif lname == 'abd_local_cross_entropy':
                    if self.inversion_replay:
                        # local classification
                        if self.cfg.increm.other_split_size == 1:
                            local_clf = lfunc(output[batch_idx,self.last_valid_out_dim:self.valid_out_dim], torch.unsqueeze((target[batch_idx]-self.last_valid_out_dim + 1).float(), dim=1), data_weights=dw_cls[batch_idx], binary=True)
                        else:
                            local_clf = lfunc(output[batch_idx,self.last_valid_out_dim:self.valid_out_dim], (target[batch_idx]-self.last_valid_out_dim).long(), dw_cls[batch_idx])
                        lval = local_clf
                        losses[lname].update(lval.item(), len(batch_idx) )
                        loss_tensors.append(lweight * lval)
                else:
                    if self.inversion_replay:
                         # ft classification  
                        with torch.no_grad():             
                            feature = self.model.forward_feature(pts_com).detach()
                        ft_clf = lfunc(self.model.final(feature)[:, :self.valid_out_dim], target_com.long(), dw_cls)
                        lval = ft_clf 
                        losses[lname].update(lval.item(), output.size(0) )
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

            output = self.model(pts)[:, :self.valid_out_dim]

            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                if lname in ['abd_distillation', 'abd_local_cross_entropy']:
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


    

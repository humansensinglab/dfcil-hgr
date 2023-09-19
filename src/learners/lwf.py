import torch
from easydict import EasyDict as edict
import torchmetrics

import optimizers as optimizer_defs
import utils
from .base import Base


class LwF(Base):

    def __init__(self, cfg, cfg_data, args, is_train, is_distributed, n_gpus):
        super(LwF, self).__init__(cfg, cfg_data, args, is_train, is_distributed, n_gpus)

    
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

            # if KD -> generate pseudo labels
            if self.KD_replay:
                allowed_predictions = list(range(self.last_valid_out_dim))
                output_hat, _ = self.previous_teacher.generate_scores(pts, allowed_predictions=allowed_predictions)
            else:
                output_hat = None

            # forward pass
            output = self.model(pts)[:, :self.valid_out_dim]

            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                if lname == 'distillation' :
                    if  output_hat is not None:
                        lval = lfunc(output, output_hat, allowed_predictions=allowed_predictions, T=self.cfg.loss.distillation.temperature, soft_t=self.cfg.loss.distillation.soft_t)
                        losses[lname].update(lval.item(), output.size(0) )
                    else:
                        losses[lname].update(0., output.size(0) )
                else:
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

            # if KD -> generate pseudo labels
            if self.KD_replay:
                allowed_predictions = list(range(self.last_valid_out_dim))
                output_hat, _ = self.previous_teacher.generate_scores(pts, allowed_predictions=allowed_predictions)
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
                'mean': acc_all,
                }, step=epoch, prefix="acc" ) 
                                 
            acc_meter.reset()   
            val_logger.flush() 

            return_values = { ltype: lmeter.avg for ltype, lmeter in losses.items() }
            return_values['acc'] = acc_all
            
            return return_values


            
    






    

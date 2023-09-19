import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import torchmetrics
import sys
import os.path as osp
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

SUB_DIR_LEVEL = 1 # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)))

import utils 


class Teacher_v1(nn.Module):

    def __init__(self, solver):

        super().__init__()
        self.solver = solver

    def generate_scores(self, x, allowed_predictions=None, threshold=None):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # set model back to its initial mode
        self.train(mode=mode)

        # threshold if desired
        if threshold is not None:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            y_hat = F.softmax(y_hat, dim=1)
            ymax, y = torch.max(y_hat, dim=1)
            thresh_mask = ymax > (threshold)
            thresh_idx = thresh_mask.nonzero().view(-1)
            y_hat = y_hat[thresh_idx]
            y = y[thresh_idx]
            return y_hat, y, x[thresh_idx]

        else:
            # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
            ymax, y = torch.max(y_hat, dim=1)

            return y_hat, y

    def generate_scores_pen(self, x):

        # set model to eval()-mode
        mode = self.training
        self.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        # set model back to its initial mode
        self.train(mode=mode)

        return y_hat



class Teacher_v2(nn.Module):

    def __init__(self, solver, sample_shape, iters, class_idx,num_inverted_class,num_known_classes, deep_inv_params, train = True, config=None):

        super().__init__()
        self.solver = solver
        self.solver.eval()
        self.sample_shape = sample_shape
        self.iters = iters
        self.config = config

        # hyperparameters
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]
        self.inv_mean = deep_inv_params[5]
        self.inv_std = deep_inv_params[6]
        

        # get class keys
        self.class_idx = list(class_idx)
        self.num_known_classes = num_known_classes
        self.num_inverted_class = num_inverted_class
        self.num_k =  self.num_inverted_class + self.num_known_classes

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()

        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in self.solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))
        self.loss_r_feature_layers = loss_r_feature_layers


    def generate_inverted_samples(self, class_size, inversion_batch_size, dataloader_batch_size, log_dir_task, inverted_sample_dir):

        # Calculate number of batches
        size = class_size * self.num_inverted_class
        num_batches = math.ceil(size / inversion_batch_size)
        num_samples_batch = inversion_batch_size
        targets_list = [i for i in range(self.num_known_classes,self.num_known_classes + self.num_inverted_class)] * class_size
        targets_list = np.array(targets_list)
        # clear cuda cache
        torch.cuda.empty_cache()
        self.solver.eval()
        self.original_pts = []
        self.inverted_pts = []
        self.inverted_targets = []
        self.inverted_outputs = []

        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes= self.config.num_total_classes).cuda()
        tb_logger = utils.TensorBoardLogger(osp.join(log_dir_task, 'inverted_samples'))
        print(f"Generating {size} inverted samples....................")
        bar = tqdm(total=num_batches, leave=True, desc='Inversion steps', dynamic_ncols=False)
        steps = 0
        for batch_id in range(num_batches):
            if batch_id == num_batches - 1:
                num_samples_batch = size - (batch_id * num_samples_batch)
            inputs = torch.normal(self.inv_mean ,self.inv_std , size=(num_samples_batch, self.sample_shape[1], self.sample_shape[2], self.sample_shape[3])).cuda()
            inputs = nn.Parameter(inputs)
            optimizer = optim.Adam([inputs], lr=self.di_lr, betas=[0.9, 0.999], eps = 1e-8)
            self.original_pts.extend(inputs.data.cpu().numpy().copy())
            targets = torch.LongTensor(targets_list[batch_id*inversion_batch_size:batch_id*inversion_batch_size + num_samples_batch]).cuda()
            acc_meter.reset()
            for iteration in range(self.iters):
                # forward with images
                self.solver.zero_grad()

                # content
                outputs = self.solver(inputs)
                loss = self.criterion(outputs / self.content_temp, targets) * self.content_weight

                # R_feature loss
                for mod in self.loss_r_feature_layers: 
                    loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
                    loss = loss + loss_distr

                optimizer.zero_grad()
                # backward pass
                loss.backward()
                with torch.no_grad():
                    # Update input image
                    optimizer.step() 
                    acc = acc_meter(outputs, targets) * 100

                if acc.item() > self.config.increm.learner.inverted_acc_thred:
                    break 

            # update tb_logger
            tb_logger.update({
                'inverted_training_loss': loss.item(),
                }, step=steps, prefix="loss" )  
            tb_logger.update({
                'inverted_training_acc': acc.item(),
                }, step=steps, prefix="acc" ) 

            steps += 1
            # update progress bar
            bar.update()
            # clear cuda cache
            torch.cuda.empty_cache()
            #print(f"Inverted stats: mean: {inputs.mean()} -- std: {inputs.std()}")
            self.inverted_pts.extend(inputs.data.cpu().numpy())
            self.inverted_targets.extend(targets.cpu().numpy())

        bar.close() 
        tb_logger.close()  
        acc_meter.reset()

        print("Check accuracy with the inverted samples:")

        # Inverted input accuracy
        num_samples_batch = inversion_batch_size
        for batch_id in range(num_batches):
            if batch_id == num_batches - 1:
                num_samples_batch = size - (batch_id * num_samples_batch)
            inputs_list = self.inverted_pts[batch_id*inversion_batch_size:batch_id*inversion_batch_size + num_samples_batch]
            targets_list = self.inverted_targets[batch_id*inversion_batch_size:batch_id*inversion_batch_size + num_samples_batch]
            inputs = torch.zeros(size=(num_samples_batch, self.sample_shape[1], self.sample_shape[2], self.sample_shape[3])).cuda()
            targets = torch.zeros(size=(num_samples_batch,)).cuda()
            for i in range(len(inputs_list)):
                inputs[i] = torch.from_numpy(inputs_list[i])
                targets[i] = targets_list[i]  
            with torch.inference_mode():
                outputs = self.solver(inputs)
                acc = acc_meter(outputs, targets) * 100
                self.inverted_outputs.extend(outputs.cpu().numpy())
        print(f"Acc inverted samples = {acc_meter.compute() * 100}")
        acc_meter.reset()

        #save inverted samples
        fname = "saved_inverted_sample" + '.pkl'
        fpath = osp.join(inverted_sample_dir, fname)
        if self.num_known_classes !=0:
            print("load saved data")
            saved_data  = utils.load_pickle(fpath);
            self.inverted_pts.extend(saved_data["pts"])
            self.inverted_targets.extend(saved_data["target"])
            self.inverted_outputs.extend(saved_data["output"])
            os.remove(fpath)

        save_dict = {
            'pts': self.inverted_pts,
            'target': self.inverted_targets,
            'output': self.inverted_outputs,
        }
        utils.save_pickle(fpath, save_dict);

        # Create a new dataset with the inverted samples
        self.inverted_dataset = InvertedDataset(self.inverted_pts, self.inverted_targets, self.inverted_outputs)
        # Create a new dataloader with the inverted dataset
        self.inverted_dataloader = DataLoader(self.inverted_dataset, batch_size=dataloader_batch_size, shuffle=True, num_workers=self.config.workers, pin_memory=True)
        # Create a iterator for the inverted dataloader
        self.inverted_iterator = iter(self.inverted_dataloader)


    def sample(self):
        # Return a batch of inverted samples
        try:
            inputs, targets, outputs = next(self.inverted_iterator)
        except StopIteration:
            self.inverted_iterator = iter(self.inverted_dataloader)
            inputs, targets, outputs = next(self.inverted_iterator)
        return inputs, targets, outputs



    def generate_scores(self, x, allowed_predictions=None, return_label=False):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        return (y, y_hat) if return_label else y_hat


    def generate_scores_pen(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
             y_hat = self.solver.forward_feature(x)

        return y_hat



class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2])
        var = input[0].permute(1, 0, 2).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        r_feature = torch.log(var**(0.5) / (module.running_var.data.type(var.type()) + 1e-8)**(0.5)).mean() - 0.5 * (1.0 - (module.running_var.data.type(var.type()) + 1e-8 + (module.running_mean.data.type(var.type())-mean)**2)/var).mean()

        self.r_feature = r_feature

            
    def close(self):
        self.hook.remove()



class InvertedDataset(Dataset):
    def __init__(self, pts, targets, outputs):
        self.pts = pts
        self.targets = targets
        self.outputs = outputs

    def __len__(self):
        return len(self.pts)

    def __getitem__(self, idx):
        x_i = self.pts[idx]
        y = np.array(self.targets[idx])
        y_hat = self.outputs[idx]
        # Convert to torch tensors
        x_i = torch.from_numpy(x_i)
        y = torch.from_numpy(y)
        y_hat = torch.from_numpy(y_hat)

        return x_i, y, y_hat
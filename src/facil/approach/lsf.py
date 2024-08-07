import importlib
from argparse import ArgumentParser

import torch
import numpy as np

from ..datasets.exemplars_dataset import ExemplarsDataset
from .incremental_learning import Inc_Learning_Appr

import facil


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning with Selective Forgetting (LSF) approach
    described in https://arxiv.org/abs/2405.18663
    """

    def __init__(self, model, device, **kwargs):
        super(Appr, self).__init__(model, device, **kwargs)
    
        # set approaches used for regularization
        appr_names = self.reg_approaches
        for appr_name in appr_names:
            # find correct class
            try:
                appr_Cls = getattr(importlib.import_module(f'.{appr_name}', package='facil.approach'), 'Appr')
            except ModuleNotFoundError:
                assert False, f'Approach {appr_name} does not exists. Make sure the argument corresponds to the file-name of the approach.'

            # use extra-args not parsed by FACIL yet
            cur_appr_args = [arg.replace(appr_name, '') for arg in facil.extra_args]
            appr_args = appr_Cls.extra_parser(cur_appr_args)[0]
            # set member, also replace names in argument witha actual approaches
            setattr(self, appr_name, appr_Cls(self.model, self.device, **facil.base_kwargs, **appr_args.__dict__))
            # we also need exemplars-dataset. this is a little weird. TODO: we could change this in init of facil   
            setattr(getattr(self, appr_name), 'exemplars_dataset', self.exemplars_dataset)

        # replace names with dict -- easier access
        self.reg_approaches = {}
        for appr_name in appr_names:
            self.reg_approaches[appr_name] = getattr(self, appr_name)

        # dict to store mnemonic codes
        self.class_codes = {}

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Eq. 1: x = lamb*x + (1-lamb)*eps "lambda is a uniform random variable in [0, 1]"
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='Mnemonic augmentation scalar (range 0-1, default=%(default)s)')
        # Eq. 5: 'lamb_sf is a balancing weight'
        parser.add_argument('--beta', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off for mnemonic codes (default=%(default)s)')
        parser.add_argument('--reg-approaches', default=[], type=str, nargs='+', required=False,
                            help='Approach(es) to use for regularization loss (default=%(default)s). Pass arguments to these approaches by their default names prepended with `approach-`, e.g. `lwf-lamb` or `lwf-T`')
        """ Pass reg-approach-args as `appr-arg`, e.g., `lwf-lamb`, `lwf-T`
        """
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def pre_train_process(self, t, trn_loader):
        
        # set optimizer for each approach. TODO: We might have to call pre-trainig here for some appr?
        for appr in self.reg_approaches:
            getattr(self, appr).optimizer = self.reg_approaches[appr]._get_optimizer()
        
        # get shape of imgas for mnemonic code
        if t == 0:
            self.class_code_shape = trn_loader.dataset.images[0].shape
            if len(self.class_code_shape) > 2:
                self.class_code_shape = (self.class_code_shape[-1], *self.class_code_shape[0:-1]) 
            self.class_code_shape = (1, *self.class_code_shape)

        for cls in np.unique(trn_loader.dataset.labels):
            self.class_codes[cls] = self.generate_class_code()

        super().pre_train_process(t, trn_loader)

    
    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # run post-training for all additional approaches used for regularization
        for appr_name in self.reg_approaches:
            # TODO: see __init__
            getattr(self, appr_name).post_train_process(t, trn_loader)


    
    def augment_samples(self, inputs, targets):
        if self.alpha == 0 or self.beta == 0:
            # no need to augment if no effect
            return None

        codes = torch.stack([self.class_codes[t.item()] for t in targets], dim=0)
        augmented_samples = self.alpha * inputs + (1 - self.alpha) * codes
        return augmented_samples

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))

            # Eq. 1, 3 -- Compute outputs of augmented samples
            augmented_outputs = None
            if self.alpha != 0 and self.beta != 0:
                augmented_samples = self.augment_samples(images, targets)
                augmented_outputs = self.model(augmented_samples.to(self.device))

            loss = self.criterion(t, outputs, targets.to(self.device), augmented_outputs)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))

                # Eq. 1, 3 -- Compute outputs of augmented samples
                augmented_outputs = None
                if self.alpha != 0 and self.beta != 0:
                    augmented_samples = self.augment_samples(images, targets)
                    augmented_outputs = self.model(augmented_samples.to(self.device))

                loss = self.criterion(t, outputs, targets.to(self.device), augmented_outputs)

                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def generate_class_code(self):
        """Returns tensor of shape `self.class_code_shape` filled with uniformly distributed values in [-1, 1)
        """
        return torch.randn(*self.class_code_shape) * 2 - 1
    
    
    def criterion(self, t, outputs, targets, augmented_outputs=None):
        """Returns the loss value"""
        c_loss = 0 # base classification loss
        m_loss = 0 # mnemonic loss (stability)
        sf_loss = 0 # selective forgetting loss (stability only on preserved classes)
        reg_loss = 0 # additional regularization

        if t > 0:
            # TODO: compute sf-loss

            # compute regularization-losses
            for appr in self.reg_approaches.values():
                reg_loss += appr.criterion(t, outputs, targets, True)[1]

        # compute mnemonic loss
        # # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            c_loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
            if augmented_outputs is not None:
                m_loss = self.beta * torch.nn.functional.cross_entropy(torch.cat(augmented_outputs, dim=1), targets)
        else:
            c_loss = torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
            if augmented_outputs is not None:
                m_loss = self.beta * torch.nn.functional.cross_entropy(augmented_outputs[t], targets - self.model.task_offset[t])

        # Eq. 2 -- loss function        
        loss = c_loss + m_loss + sf_loss + reg_loss
        return loss
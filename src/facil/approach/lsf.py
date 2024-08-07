import importlib
import itertools
from argparse import ArgumentParser

import torch

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


    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Eq. 1: x = lamb*x + (1-lamb)*eps "lambda is a uniform random variable in [0, 1]"
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='Mnemonic augmentation trade-off (default=%(default)s')
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
        
        super().pre_train_process(t, trn_loader)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # run post-training for all additional approaches used for regularization
        for appr_name in self.reg_approaches:
            # TODO: see __init__
            getattr(self, appr_name).post_train_process(t, trn_loader)

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = 0

        if t > 0:
            # compute sf-loss TODO

            # compute regularization-losses
            reg_loss = 0
            for appr in self.reg_approaches.values():
                reg_loss += appr.criterion(t, outputs, targets, True)[1]

            loss += reg_loss
        # if t > 0:
        #     loss_reg = 0
        #     # Eq. 3: elastic weight consolidation quadratic penalty
        #     for n, p in self.model.model.named_parameters():
        #         if n in self.fisher.keys():
        #             loss_reg += torch.sum(self.fisher[n] * (p - self.older_params[n]).pow(2)) / 2
        #     loss += self.lamb * loss_reg
        # # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

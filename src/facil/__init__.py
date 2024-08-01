import os
import sys
import argparse
import importlib
import time

from functools import reduce

import torch

import numpy as np

from . import approach
from .approach.incremental_learning import Inc_Learning_Appr
from .approach.finetuning import Appr as Appr_finetuning
from .loggers.exp_logger import MultiLogger
from .datasets.data_loader import get_loaders
from .datasets.dataset_config import dataset_config
from .datasets.exemplars_dataset import ExemplarsDataset
from .last_layer_analysis import last_layer_analysis
from .networks import tvmodels, allmodels, set_tvmodel_head_var
from .networks.network import LLL_Net
from . import utils
from .gridsearch import GridSearch

def init():
    """ Perform all initialization steps previously done in `main_incremental.py` and stop before training.
    """
    global tstart
    tstart = time.time()

    # load basic configuration
    load_args()
    load_exp_name()
    init_cuda()

    print_args()
    
    init_logger()
    # add logger to base-args for approach constructor
    base_kwargs['logger'] = logger

    # we can now start logging
    log_args()

    # initialize model, data, approach
    utils.seed_everything()
    init_model()
    utils.seed_everything()
    load_data()
    utils.seed_everything()
    init_approach()
    utils.seed_everything()

def train():
    from . import training
    return training.train()

def load_base_args():
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=4, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=[0.1], nargs='+',  type=float,
                        help='List of starting learning rates for each task. If length is less than number of tasks, will be filled with last learning-rate. If length is greather than number of tasks, will be stripped. (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    # gridsearch args
    parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                        help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')

    # Args -- Incremental Learning Framework
    global args, extra_args
    args, extra_args = parser.parse_known_args(sys.argv)
    args.results_path = os.path.expanduser(args.results_path)

    # extend/strip learning-rate list to numer of tasks
    if len(args.lr) < args.num_tasks:
        args.lr.extend([args.lr[-1] for _ in range(args.num_tasks - len(args.lr))])
    elif len(args.lr) > args.num_tasks:
        args.lr = args.lr[:args.num_tasks]

    global base_kwargs
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)


def load_approach_args():
    # Args -- Continual Learning Approach
    global Appr, appr_args, extra_args

    Appr = getattr(importlib.import_module(name='.approach.{}'.format(args.approach), package='facil'), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)


def load_exemplar_args():
    global Appr_ExemplarsDataset, appr_exemplars_dataset_args, extra_args

    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()


def load_gridsearch_args():
    global GridSearch_ExemplarsDataset, gs_args, extra_args

    gs_args, extra_args = GridSearch.extra_parser(extra_args)
    assert issubclass(Appr_finetuning, Inc_Learning_Appr)
    GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()


def load_args():
    load_base_args()
    load_approach_args()
    load_exemplar_args()
    load_gridsearch_args()


def print_args():
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    if Appr_ExemplarsDataset:
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)

    if args.gridsearch_tasks > 0:
        print('GridSearch arguments =')
        for arg in np.sort(list(vars(gs_args).keys())):
            print('\t' + arg + ':', getattr(gs_args, arg))
        print('=' * 108)

def log_args():
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__, **gs_args.__dict__))

def load_data():
    global trn_loader, val_loader, tst_loader, taskcla, max_task

    utils.seed_everything(seed=args.seed)

    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task


def init_approach():
    global appr_kwargs, appr, gridsearch

    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **appr_args.__dict__,}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)

    utils.seed_everything(seed=args.seed)

    appr = Appr(net, device, **appr_kwargs)

    # init gridsearch
    if args.gridsearch_tasks > 0:
        ft_kwargs = dict(**base_kwargs, exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices), all_outputs=False)
        appr_ft = Appr_finetuning(net, device, **ft_kwargs)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)


def init_model():
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        if args.network == 'googlenet':
            init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        base_net = getattr(importlib.import_module(name='.networks', package='facil'), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = base_net(pretrained=False)

    global net
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)


def init_cuda():
    if args.no_cudnn_deterministic:
        logger.log_print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    global device

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        logger.log_logger.log_print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'


def load_exp_name():
    global full_exp_name
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name


def init_logger():
    global logger
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)

import os
import torch
import random
import numpy as np

cudnn_deterministic = True

import facil

def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        logger.log_print('*' * 108)
        logger.log_print(name)
        for i in range(metric.shape[0]):
            logger.log_print('\t', end='')
            for j in range(metric.shape[1]):
                logger.log_print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    logger.log_print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                logger.log_print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            logger.log_print()
    logger.log_print('*' * 108)

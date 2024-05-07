import os
import math
import time
import random
import yaml
import numpy as np
import torch
import torch.distributed as dist


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dist_init():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    host_ip = os.environ['MASTER_ADDR']
    port = os.environ['MASTER_PORT']
    init_method = 'tcp://{}:{}'.format(host_ip, port)
    print('dist.init_process_group', init_method, world_size, rank)
    dist.init_process_group('nccl', init_method=init_method, world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)
    print('DDP: rank is {}, local_rank is {}, world_size is {}, host ip is {}'.format(rank, local_rank, world_size, host_ip))
    return local_rank


class Logger:
    
    def __init__(self, ):
        self.level = 'Info'

    def set_level(self, level):
        self.level = level

    def info(self, *args):
        if self.level == 'None':
            return
        elif self.level == 'Info':
            print('[' + time.asctime(time.localtime(time.time())) + ']', *args)


class FakeRecorder:
    
    '''
    This recorder will do nothing
    '''
    def add_scalar(*args, **kwargs):
        pass


def merge_configs(configs, base_configs):
    for key in base_configs:
        if not key in configs:
            configs[key] = base_configs[key]
        elif type(configs[key]) is dict:
            merge_configs(configs[key], base_configs[key])


def build_configs(config_file, loaded_config_files):
    loaded_config_files.append(config_file)
    with open(config_file, 'r') as reader:
        configs = yaml.load(reader, Loader=yaml.Loader)
    for base_config_file in configs['base']:
        base_config_file = os.getcwd() + '/configs/' + base_config_file
        if base_config_file in loaded_config_files:
            continue
        base_configs = build_configs(base_config_file, loaded_config_files)
        merge_configs(configs, base_configs)
    return configs


def clear_configs(configs):
    keys = list(configs.keys())
    for key in keys:
        if type(configs[key]) is dict:
            configs[key] = clear_configs(configs[key])
        elif configs[key] == 'None':
            print('Clear config', key)
            configs.pop(key)
    return configs


def get_lr_schedule(num_warmup_iters, num_cosine_iters, cosine_times, start_lr, peak_lr, end_lr):
    '''
    Params:
        num_warmup_iters: number of warmup iterations
        num_cosine_iters: number of cosine annealing iterations
        cosine_times: how many times we will do cosine annealing
        start_lr: initial learning rate of the warmup
        peak_lr: final learning rate of the warmup, initial learning rate of the cosine annealing
        end_lr: final learning rate of the cosine annealing
    Retrun:
        lr_schedule: [lr_0, lr_1, ..., lr_{n-1}], n is the total number of iterations
    '''
    warmup_lr_schedule = np.linspace(start_lr, peak_lr, num_warmup_iters)
    iters = np.arange(num_cosine_iters)
    cosine_lr_schedule = np.array(
        [
            end_lr + (
                0.5 * (peak_lr - end_lr)
                * (1 + math.cos(math.pi * t / num_cosine_iters))
            )
            for t in iters
        ]
    )
    lr_schedule = np.concatenate([warmup_lr_schedule] + [cosine_lr_schedule] * cosine_times)
    return lr_schedule

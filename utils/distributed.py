import pickle

import torch
from torch import distributed as dist


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()

def get_device():
    if not dist.is_available():
        return torch.device('cuda')

    if not dist.is_initialized():
        return torch.device('cuda')

    return torch.device('cuda', dist.get_rank())

def synchronize(local_rank=0):
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()

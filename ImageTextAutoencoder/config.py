from __future__ import division
from __future__ import print_function
import torch
import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

encoding_dim = 1024 #

# Dataset name: flowers, birds
__C.DATASET_NAME = 'birds'#'flowers' #'birds'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''

__C.GPU_ID = '0'
__C.CUDA = torch.cuda.is_available()

__C.WORKERS = 6

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3 # change this to 1 if planning to use GNET() for only one branch
__C.TREE.BASE_SIZE = 64


# Test options
__C.TEST = edict()
__C.TEST.B_EXAMPLE = True
__C.TEST.SAMPLE_NUM = 30000


# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.VIS_COUNT = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4#2e-4
__C.TRAIN.GENERATOR_LR = 6e-4#2e-4
__C.TRAIN.IMAGE_TEXT_UPDATES = 1
__C.TRAIN.ADV_UPDATES = 3
__C.TRAIN.FLAG = True
# __C.TRAIN.DPP = True
__C.TRAIN.NET_G = ''
__C.TRAIN.NET_D = ''

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0
__C.TRAIN.COEFF.UNCOND_LOSS = 1.0 #make it 0 when not using embedding in DNETs
__C.TRAIN.COEFF.COLOR_LOSS = 0.0


# Modal options
__C.GAN = edict()
__C.GAN.EMBEDDING_DIM = 128
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 64
__C.GAN.Z_DIM = 100
__C.GAN.NETWORK_TYPE = 'default'
__C.GAN.R_NUM = 2
__C.GAN.B_CONDITION = True # make it false if no encoding is provided to discriminator



# IMAGE options
__C.IMAGE = edict()
__C.IMAGE.ENCODER_NAME = "resnet"
__C.IMAGE.FIX_ENCODER = True
__C.IMAGE.PRETRAINED_ENCODER = True
__C.IMAGE.DIMENSION = encoding_dim
__C.IMAGE.NR_UPDATES = 1
__C.IMAGE.VAE = True
__C.IMAGE.DPP = True
__C.IMAGE.ENCDPP = False
__C.IMAGE.ADV = edict()
__C.IMAGE.ADV.LAYERS = 3
__C.IMAGE.ADV.DIM = encoding_dim
__C.IMAGE.ADV.DROPOUT = 0.3
__C.IMAGE.ADV.INPUT_DROPOUT = 0.3
__C.IMAGE.ADV.NOISE = True

# Text options
__C.TEXT = edict()
__C.TEXT.DIMENSION = 1024  #1024 input dimention to GNET()
__C.TEXT.N_LAYERS = 2
__C.TEXT.NR_UPDATES = 1
__C.TEXT.EMBEDDING_DIM = 100
__C.TEXT.VAE = True
__C.TEXT.ENCDPP = False
__C.TEXT.DROPOUT = 0.0
__C.TEXT.TEACHER_FORCING = 0.0
__C.TEXT.ADV = edict()
__C.TEXT.ADV.LAYERS = 3
__C.TEXT.ADV.DIM = encoding_dim
__C.TEXT.ADV.DROPOUT = 0.3
__C.TEXT.ADV.INPUT_DROPOUT = 0.3
__C.TEXT.ADV.NOISE = True


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

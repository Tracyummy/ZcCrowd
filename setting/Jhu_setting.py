from easydict import EasyDict as edict


# init
__C_jhu = edict()

cfg_data = __C_jhu

__C_jhu.TRAIN_SIZE = (512,1024)
__C_jhu.VAL4EVAL = 'val_gt_loc.txt'

__C_jhu.MEAN_STD = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])

__C_jhu.LABEL_FACTOR = 1
__C_jhu.LOG_PARA = 1.

__C_jhu.RESUME_MODEL = ''#model path
__C_jhu.TRAIN_BATCH_SIZE = 6 #imgs

__C_jhu.VAL_BATCH_SIZE = 1 # must be 1
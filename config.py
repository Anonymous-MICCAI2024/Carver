
import os
import json
import glob

from utils import get_path_with_annotation,get_path_with_annotation_ratio,get_path_with_column
from utils import get_weight_path


__3d_net__ = ['unet_3d', 'se_res_unet', 'da_unet','da_se_unet','res_da_se_unet', 'UNETR', 'vnet', 'vnet_res', 'vnet_lite', 'vnet_se', 'vnet_da', 'vnet_res_se']
__mode__ = ['3d']


DISEASE = 'cardio' 
MODE = '3d'
NET_NAME = 'vnet'
ENCODER_NAME = 'simplenet'
VERSION = 'v2_r0.9.2'

DEVICE = '1'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = True
# True if use external pre-trained model 
EX_PRE_TRAINED = True if 'pretrain' in VERSION else False
# True if use resume model
CKPT_POINT = False

FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 5
GPU_NUM = len(DEVICE.split(','))

if DISEASE != 'cardio':
    with open(json_path[DISEASE], 'r') as fp:
        info = json.load(fp)
else:
    info = None

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = None# or 1,2,...
NUM_CLASSES = 2# info['annotation_num'] + 1 # 2 for binary, more for multiple classes
if ROI_NUMBER is not None:
    if isinstance(ROI_NUMBER,list):
        NUM_CLASSES = len(ROI_NUMBER) + 1
        ROI_NAME = 'Part_{}'.format(str(len(ROI_NUMBER)))
    else:
        NUM_CLASSES = 2
        ROI_NAME = info['annotation_list'][ROI_NUMBER - 1]
else:
    ROI_NAME = 'All'


try:
    SCALE = info['scale'][ROI_NAME]
    MEAN = STD = None
except:
    # SCALE = None
    # MEAN = info['mean_std']['mean']
    # STD = info['mean_std']['std']
    SCALE = MEAN = STD = None
#---------------------------------

#--------------------------------- mode and data path setting

if MODE == '3d':
    PATH_LIST = cardio_3d__profile_points_list_path
#---------------------------------


#--------------------------------- others
INPUT_SHAPE = (2, 128, 128 ,128) if MODE =='3d' 
BATCH_SIZE = 24 if MODE =='3d' 


CKPT_PATH = './new_ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

INIT_TRAINER = {
  'net_name':NET_NAME,
  'encoder_name':ENCODER_NAME,
  'lr':1e-3, 
  'n_epoch':120,
  'channels':1,
  'num_classes':NUM_CLASSES, 
  'roi_number':ROI_NUMBER, 
  'scale':SCALE,
  'input_shape':INPUT_SHAPE,
  'crop':0,
  'batch_size':BATCH_SIZE,
  'num_workers':2,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'ex_pre_trained':EX_PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'use_moco':None if 'moco' not in VERSION else 'moco',
  'weight_decay': 0.0001,
  'momentum': 0.9,
  'gamma': 0.1,
  'milestones': [30,60,90],
  'T_max':5,
  'mean':MEAN,
  'std':STD,
  'topk':20,
  'use_fp16':True, #False if the machine you used without tensor core
 }
#---------------------------------

__loss__ = ['TopKLoss','DiceLoss','CEPlusDice','CELabelSmoothingPlusDice','OHEM','Cross_Entropy']
# Arguments when perform the trainer 
loss_index = 0 if len(VERSION.split('.')) == 2 else eval(VERSION.split('.')[-1].split('-')[0])
LOSS_FUN = 'TopkCEPlusDice' if ROI_NUMBER is not None else __loss__[loss_index]

print('>>>>> loss fun:%s'%LOSS_FUN)

SETUP_TRAINER = {
  'output_dir':'./new_ckpt/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME),
  'log_dir':'./new_log/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME),
  'optimizer':'AdamW',
  'loss_fun':LOSS_FUN,
  'class_weight':None,
  'lr_scheduler':'CosineAnnealingLR', #'CosineAnnealingLR','CosineAnnealingWarmRestarts''MultiStepLR'
  'freeze_encoder':False,
  'get_roi': False if 'roi' not in VERSION else True
}
#---------------------------------

TEST_PATH = cardio_3d__profile_points_list_path


        

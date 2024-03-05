
import os
import json
import glob

from utils import get_path_with_annotation,get_path_with_annotation_ratio,get_path_with_column
from utils import get_weight_path


__disease__ = ['Cervical','Nasopharynx','Structseg_HaN','Structseg_THOR','SegTHOR','Lung']
# __2d_net__ = ['unet_2d']  old ./ckpt
__2d_net__ = ['unet','unet++','FPN','deeplabv3+','att_unet','res_unet']
__trans_net__ = ['UTNet','UTNet_encoder','TransUNet','ResNet_UTNet','SwinUNet']
__new_net__ = ['sfnet']
__encoder_name__ = ['simplenet','resnet18','resnet34','resnet50','se_resnet50', \
                   'resnext50_32x4d','timm-resnest14d','timm-resnest26d','timm-resnest50d', \
                    'efficientnet-b4', 'efficientnet-b5']

__new_encoder_name__ = ['swin_transformer','swinplusr18']
__3d_net__ = ['unet_3d', 'se_res_unet', 'da_unet','da_se_unet','res_da_se_unet', 'UNETR', 'vnet', 'vnet_res', 'vnet_lite', 'vnet_se', 'vnet_da', 'vnet_res_se']
__mode__ = ['2d','2d_clean','3d']

json_path = {
    'Cervical':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Cervical_Oar.json',
    'Nasopharynx':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Nasopharynx_Oar.json',
    'Liver':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Liver_Oar.json',
    'Stomach':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Stomach_Oar.json',
    'Structseg_HaN':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Structseg_HaN.json',
    'Structseg_THOR':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Structseg_THOR.json',
    'HaN_GTV':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/HaN_GTV.json',
    'THOR_GTV':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/THOR_GTV.json',
    'SegTHOR':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/SegTHOR.json',
    'Covid-Seg':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/Covid-Seg.json', # competition
    'Lung':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Lung_Oar.json',
    'Lung_Tumor':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Lung_Tumor.json',
    'Nasopharynx_Tumor':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Nasopharynx_Tumor.json',
    'Cervical_Tumor':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/Cervical_Tumor.json',
    'EGFR':'/staff/shijun/torch_projects/Med_Seg/converter/dcm_converter/static_files/EGFR.json',
    'LITS':'/staff/shijun/torch_projects/Med_Seg/converter/nii_converter/static_files/LITS.json', # competition
}


DISEASE = 'kbr' 
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

if DISEASE != 'kbr':
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
kbr_3d_convexhull_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_points_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_convexhull_list_path = glob.glob(kbr_3d_convexhull_data_path+ '/*.hdf5')

kbr_3d_points_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_points_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_points_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_points_list_path = glob.glob(kbr_3d_points_data_path+ '/*.hdf5')

kbr_3d_poly_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_poly_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_points_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_poly_list_path = glob.glob(kbr_3d_poly_data_path+ '/*.hdf5')

kbr_3d_profile_points_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d__profile_points_list_path = glob.glob(kbr_3d_profile_points_data_path+ '/*.hdf5')

kbr_3d_profile_points_remove_1_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_profile_points_remove_1_list_path = glob.glob(kbr_3d_profile_points_remove_1_data_path+ '/*.hdf5')

kbr_3d_profile_points_remove_2_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_2_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_profile_points_remove_2_list_path = glob.glob(kbr_3d_profile_points_remove_2_data_path+ '/*.hdf5')

kbr_3d_profile_points_remove_3_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_3_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_profile_points_remove_3_list_path = glob.glob(kbr_3d_profile_points_remove_3_data_path+ '/*.hdf5')

kbr_3d_profile_points_remove_4_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_4_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_profile_points_remove_4_list_path = glob.glob(kbr_3d_profile_points_remove_4_data_path+ '/*.hdf5')

kbr_3d_profile_points_remove_5_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_5_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_profile_points_remove_5_list_path = glob.glob(kbr_3d_profile_points_remove_5_data_path+ '/*.hdf5')

kbr_3d_profile_points_remove_6_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_6_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_profile_points_remove_6_list_path = glob.glob(kbr_3d_profile_points_remove_6_data_path+ '/*.hdf5')

kbr_3d_profile_points_remove_7_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_7_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_profile_points_remove_7_list_path = glob.glob(kbr_3d_profile_points_remove_7_data_path+ '/*.hdf5')

kbr_3d_profile_points_remove_8_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_8_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_1_128_backup' # '/staff/ydli/projects/OReX/Data/UNet/hdf5_backup' 
kbr_3d_profile_points_remove_8_list_path = glob.glob(kbr_3d_profile_points_remove_8_data_path+ '/*.hdf5')

kbr_3d_profile_lines_data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_line_128_backup'
kbr_3d_profile_lines_list_path = glob.glob(kbr_3d_profile_lines_data_path+ '/*.hdf5')

kbr_ed_path = kbr_3d_convexhull_data_path + '/*ed.hdf5'
kbr_ed_data_list = glob.glob(kbr_ed_path)    # os.dirs

kbr_es_path = kbr_3d_convexhull_data_path + '/*es.hdf5'
kbr_es_data_list = glob.glob(kbr_es_path)    # os.dirs

if MODE == '2d_clean':
    assert ROI_NUMBER is not None, "roi number must not be None in 2d clean"
    PATH_LIST = get_path_with_annotation(info['2d_data']['csv_path'],'path',ROI_NAME)
elif MODE == '2d':
    PATH_LIST = glob.glob(os.path.join(info['2d_data']['save_path'],'*.hdf5'))
    # PATH_LIST = get_path_with_annotation_ratio(info['2d_data']['csv_path'],'path',ROI_NAME,ratio=0.5)
else:
    PATH_LIST = kbr_3d__profile_points_list_path
#---------------------------------


#--------------------------------- others
INPUT_SHAPE = (128, 128 ,128) if MODE =='3d' else (512,512)#(512,512)
BATCH_SIZE = 24 if MODE =='3d' else 32


# CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format('Covid-Seg',MODE,'v1.0','Lesion',str(CURRENT_FOLD))
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

TEST_PATH = kbr_3d__profile_points_list_path


if DISEASE in ['Cervical','Nasopharynx','Lung','Lung_Tumor']:
    if MODE == '2d_clean':
        TEST_PATH = get_path_with_annotation(info['2d_data']['test_csv_path'],'path',ROI_NAME)
    elif MODE == '2d':
        TEST_PATH = get_path_with_column(info['2d_data']['test_csv_path'],'path')

        
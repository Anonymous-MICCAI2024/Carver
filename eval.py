import os 
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils.data_loader import DataGenerator, To_Tensor, CropResize, Trunc_and_Normalize
from data_utils.transformer_2d import Get_ROI
from torch.cuda.amp import autocast as autocast
from utils import get_weight_path,multi_dice,multi_hd


def resize_and_pad(pred,true,num_classes,target_shape,bboxs):
    from skimage.transform import resize
    final_pred = []
    final_true = []

    for bbox, pred_item, true_item in zip(bboxs,pred,true):
        h,w = bbox[2]-bbox[0], bbox[3]-bbox[1]
        new_pred = np.zeros(target_shape,dtype=np.float32)
        new_true = np.zeros(target_shape,dtype=np.float32)
        for z in range(1,num_classes):
            roi_pred = resize((pred_item == z).astype(np.float32),(h,w),mode='constant')
            new_pred[bbox[0]:bbox[2],bbox[1]:bbox[3]][roi_pred>=0.5] = z
            roi_true = resize((true_item == z).astype(np.float32),(h,w),mode='constant')
            new_true[bbox[0]:bbox[2],bbox[1]:bbox[3]][roi_true>=0.5] = z
        final_pred.append(new_pred)
        final_true.append(new_true)
    
    final_pred = np.stack(final_pred,axis=0)
    final_true = np.stack(final_true,axis=0)
    return final_pred, final_true

def get_net(net_name,encoder_name,channels=1,num_classes=2,input_shape=(512,512)):

        if net_name == 'unet':
            if encoder_name in ['simplenet','swin_transformer','swinplusr18']:
                from model.unet import unet
                net = unet(net_name,encoder_name=encoder_name,in_channels=channels,classes=num_classes,aux_classifier=True)
            else:
                import segmentation_models_pytorch as smp
                net = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=channels,
                    classes=num_classes,                     
                    aux_params={"classes":num_classes-1} 
                )
        elif net_name == 'unet++':
            if encoder_name is None:
                raise ValueError(
                    "encoder name must not be 'None'!"
                )
            else:
                import segmentation_models_pytorch as smp
                net = smp.UnetPlusPlus(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=channels,
                    classes=num_classes,                     
                    aux_params={"classes":num_classes-1} 
                )

        elif net_name == 'FPN':
            if encoder_name is None:
                raise ValueError(
                    "encoder name must not be 'None'!"
                )
            else:
                import segmentation_models_pytorch as smp
                net = smp.FPN(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=channels,
                    classes=num_classes,                     
                    aux_params={"classes":num_classes-1} 
                )
        
        elif net_name == 'deeplabv3+':
            if encoder_name is None:
                raise ValueError(
                    "encoder name must not be 'None'!"
                )
            else:
                import segmentation_models_pytorch as smp
                net = smp.DeepLabV3Plus(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=channels,
                    classes=num_classes,                     
                    aux_params={"classes":num_classes-1} 
                )
        
        elif net_name == 'res_unet':
            from model.res_unet import res_unet
            net = res_unet(net_name,encoder_name=encoder_name,in_channels=channels,classes=num_classes)
        
        elif net_name == 'att_unet':
            from model.att_unet import att_unet
            net = att_unet(net_name,encoder_name=encoder_name,in_channels=channels,classes=num_classes)
        
        elif net_name == 'sfnet':
            from model.sfnet import sfnet
            net = sfnet(net_name,encoder_name=encoder_name,in_channels=channels,classes=num_classes)
        
        ## external transformer + Unet
        elif net_name == 'UTNet':
            from model.trans_model.utnet import UTNet
            net = UTNet(channels, base_chan=32,num_classes=num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
        elif net_name == 'UTNet_encoder':
            from model.trans_model.utnet import UTNet_Encoderonly
            # Apply transformer blocks only in the encoder
            net = UTNet_Encoderonly(channels, base_chan=32, num_classes=num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
        elif net_name =='TransUNet':
            from model.trans_model.transunet import VisionTransformer as ViT_seg
            from model.trans_model.transunet import CONFIGS as CONFIGS_ViT_seg
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = num_classes 
            config_vit.n_skip = 3 
            config_vit.patches.grid = (int(input_shape[0]/16), int(input_shape[1]/16))
            net = ViT_seg(config_vit, img_size=input_shape[0], num_classes=num_classes)
            #net.load_from(weights=np.load('./initmodel/R50+ViT-B_16.npz')) # uncomment this to use pretrain model download from TransUnet git repo

        elif net_name == 'ResNet_UTNet':
            from model.trans_model.resnet_utnet import ResNet_UTNet
            net = ResNet_UTNet(channels, num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True)
        
        elif net_name == 'SwinUNet':
            from model.trans_model.swin_unet import SwinUnet, SwinUnet_config
            config = SwinUnet_config()
            config.num_classes = num_classes
            config.in_chans = channels
            net = SwinUnet(config, img_size=input_shape[0], num_classes=num_classes)
        
        return net


def eval_process(test_path,config):
    # data loader
    test_transformer = transforms.Compose([
                Trunc_and_Normalize(config.scale),
                Get_ROI(pad_flag=False) if config.get_roi else transforms.Lambda(lambda x:x),
                CropResize(dim=config.input_shape,num_class=config.num_classes,crop=config.crop),
                To_Tensor(num_class=config.num_classes)
            ])

    test_dataset = DataGenerator(test_path,
                                roi_number=config.roi_number,
                                num_class=config.num_classes,
                                transform=test_transformer)

    test_loader = DataLoader(test_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True)
    
    # get weight
    weight_path = get_weight_path(config.ckpt_path)
    print(weight_path)

    # get net
    net = get_net(config.net_name,config.encoder_name,config.channels,config.num_classes,config.input_shape)
    checkpoint = torch.load(weight_path,map_location='cpu')
    msg=net.load_state_dict(checkpoint['state_dict'],strict=False)
    print(msg)
    pred = []
    true = []
    net = net.cuda()
    net.eval()

    with torch.no_grad():
        for step, sample in enumerate(test_loader):
            data = sample['image']
            target = sample['label']

            data = data.cuda()

            with autocast(False):
                output = net(data)
            if isinstance(output,tuple) or isinstance(output,list):
                seg_output = output[0]
            else:
                seg_output = output
            seg_output = torch.argmax(torch.softmax(seg_output, dim=1),1).detach().cpu().numpy()                          
            target = torch.argmax(target,1).detach().cpu().numpy()
            if config.get_roi:
                bboxs = torch.stack(sample['bbox'],dim=0).cpu().numpy().T
                seg_output,target = resize_and_pad(seg_output,target,config.num_classes,config.input_shape,bboxs)
            pred.append(seg_output)
            true.append(target)
    pred = np.concatenate(pred,axis=0)
    true = np.concatenate(true,axis=0)

    return pred,true


class Config:
    num_classes_dict = {
        'Cervical':7,
        'Lung':5,
        'Nasopharynx':19,
        'Liver':9,
        'Stomach':10
    }
    scale_dict = {
        'Cervical':[-200,400],
        'Lung':[-800,400],
        'Nasopharynx':[-150,250],
        'Liver':[-200,400],
        'Stomach':[-200,400]
    }
    
    input_shape = (512,512)#(448,448)
    channels = 1
    crop = 0
    roi_number = None

    disease = 'Lung'
    num_classes = num_classes_dict[disease]
    scale = scale_dict[disease]
    net_name = 'res_unet'
    encoder_name = 'simplenet'
    version = 'v6.0-roi'
    get_roi = False if 'roi' not in version else True
    fold = 1
    ckpt_path = f'./new_ckpt/{disease}/2d/{version}/All/fold{str(fold)}'


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # test data
    data_path_dict = {
        'Cervical':'/staff/shijun/dataset/Med_Seg/Cervical_Oar/2d_test_data',
        'Lung':'/staff/shijun/dataset/Med_Seg/Lung_Oar/2d_test_data',
        'Nasopharynx':'/staff/shijun/dataset/Med_Seg/Nasopharynx_Oar/2d_test_data',
        'Liver':'/staff/shijun/dataset/Med_Seg/Liver_Oar/2d_test_data',
        'Stomach':'/staff/shijun/dataset/Med_Seg/Stomach_Oar/2d_test_data'
    }
    excpet_dict = {
        'Cervical':[],
        'Lung':[],
        'Nasopharynx':[],
        'Liver':[],
        'Stomach':[]
    }
    start = time.time()
    config = Config()
    data_path = data_path_dict[config.disease]
    sample_list = list(set([case.name.split('_')[0] for case in os.scandir(data_path)]))
    sample_list.sort()
    except_list = excpet_dict[config.disease]
    sample_list = [sample for sample in sample_list if sample not in except_list]
    for fold in range(1,6):
        print('>>>>>>>>>>>>fold%d>>>>>>>>>>>>'%fold)
        total_dice = []
        total_hd = []
        info_dice = []
        info_hd = []
        config.fold = fold
        config.ckpt_path = f'./new_ckpt/{config.disease}/2d/{config.version}/All/fold{str(fold)}'
        for sample in sample_list:
            info_item_dice = []
            info_item_hd = []
            info_item_dice.append(sample)
            info_item_hd.append(sample)
            print('>>>>>>>>>>>>%s is being processed'%sample)
            test_path = [case.path for case in os.scandir(data_path) if case.name.split('_')[0] == sample]
            test_path.sort(key=lambda x:eval(x.split('_')[-1].split('.')[0]))
            print(len(test_path))
            pred,true = eval_process(test_path,config)

            category_dice, avg_dice = multi_dice(true,pred,config.num_classes - 1)
            total_dice.append(category_dice)
            print('category dice:',category_dice)
            print('avg dice: %s'% avg_dice)

            category_hd, avg_hd = multi_hd(true,pred,config.num_classes - 1)
            total_hd.append(category_hd)
            print('category hd:',category_hd)
            print('avg hd: %s'% avg_hd)

            info_item_dice.extend(category_dice)
            info_item_hd.extend(category_hd)

            info_dice.append(info_item_dice)
            info_hd.append(info_item_hd)

        dice_csv = pd.DataFrame(data=info_dice)
        hd_csv = pd.DataFrame(data=info_hd)
        if not os.path.exists(f'./result/raw_data/{config.disease}'):
            os.makedirs(f'./result/raw_data/{config.disease}')
        dice_csv.to_csv(f'./result/raw_data/{config.disease}/{config.version}_fold{config.fold}_dice.csv')
        hd_csv.to_csv(f'./result/raw_data/{config.disease}/{config.version}_fold{config.fold}_hd.csv')

        total_dice = np.stack(total_dice,axis=0) #sample*classes
        total_category_dice = np.mean(total_dice,axis=0)
        total_avg_dice = np.nanmean(total_category_dice)

        print('total category dice mean:',total_category_dice)
        print('total category dice std:',np.std(total_dice,axis=0))
        print('total dice mean: %s'% total_avg_dice)


        total_hd = np.stack(total_hd,axis=0) #sample*classes
        total_category_hd = np.mean(total_hd,axis=0)
        total_avg_hd = np.nanmean(total_category_hd)

        print('total category hd mean:',total_category_hd)
        print('total category hd std:',np.std(total_hd,axis=0))
        print('total hd mean: %s'% total_avg_hd)

        print("runtime:%.3f"%(time.time() - start))
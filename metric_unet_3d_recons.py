from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
import numpy as np
import os
import torch
import glob
import pandas as pd
import h5py

from xml_reader import parseXml
import trimesh
from trimesh.voxel import creation
from trimesh import Trimesh

from kbr_slice import read_kbr_plane, read_kbr_mesh

from skimage.metrics import hausdorff_distance
from utils_seg import cal_score, cal_asd

from skimage import measure
import copy
import SimpleITK as sitk

def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=int)
    hdf5_file.close()

    return image

def read_obj_file(pred_file):
    obj = trimesh.load(pred_file)

    # print(type(obj))
    verts = obj.vertices
    # faces = obj.faces

    voxels = creation.local_voxelize(obj, obj.centroid, pitch=0.005, radius=128, fill=True)
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)
    # print(matrix.shape)

    return verts, matrix

def read_xml_file(gt_name, mode, sid):

    info_list, cali_info, mesh_text = parseXml(gt_name)
    verts, faces = read_kbr_mesh(mesh_text[mode])
    verts -= np.mean(verts, axis=0)
    scale = 1.1
    verts /= scale * np.max(np.absolute(verts))

    hdf5_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5/'
    hdf5_file = glob.glob(hdf5_path+'/*'+sid+'.hdf5')[0]
    # print(hdf5_file)
    matrix = hdf5_reader(hdf5_file,'image')

    return verts, matrix


def metric_recons_hd(pred_matrix, gt_matrix):
    '''
    Args:
        pred_points, gt_points: np arrays
    Returns:
          hausdorff distance
    '''

    hausdorff_d = hausdorff_distance(pred_matrix, gt_matrix)

    return hausdorff_d

def metric_recons_cd(pred_points, gt_points):
    '''
    Args:
        pred_points, gt_points: np arrays
    Returns:
        chamfer distance
    '''

    tree = KDTree(gt_points)
    dist_pred_points = tree.query(pred_points)[0]
    tree = KDTree(pred_points)
    dist_gt_points = tree.query(gt_points)[0]

    return np.mean(dist_pred_points) + np.mean(dist_gt_points)

def metric_recons_IOU(preds_matrix, gt_matrix):
    '''
    Args:
		preds (np.array) voxels
		gt (np,array) voxels

	Returns:
		float: IoU
    '''
    intersec = np.logical_and(preds_matrix, gt_matrix)
    union = np.logical_or(preds_matrix, gt_matrix)
    voxels_IOU = np.sum(intersec) / np.sum(union)

    return voxels_IOU


def metric_recons_DICE(preds_matrix, gt_matrix):
    '''
    Args:
		preds (np.array) voxels
		gt (np,array) voxels

	Returns:
		float: DICE
    '''
    count_pred_true = np.sum(preds_matrix)
    count_gt_true = np.sum(gt_matrix)
    intersec = np.logical_and(preds_matrix, gt_matrix)
    voxels_DICE = 2*np.sum(intersec) / (count_pred_true+count_gt_true)


    return voxels_DICE

def get_sid_efs(sid ,ed_pred_volume, ed_gt_volume, datapath):
    sid = sid.split('_')[2]
    sid = '3042_'+sid

    file = 'SID'+'_'+sid+'_es.hdf5'
    print('es file:', file)
    file_path = datapath + '/'+file

    if os.path.exists(file_path):
        hdf5_file = h5py.File(file_path, 'r')
        data_voxel = np.asarray(hdf5_file['data'])
        preds_matrix = np.asarray(hdf5_file['inf_image'])
        gt_matrix = np.asarray(hdf5_file['gt_label'])

        es_pred_volume = np.sum(preds_matrix)
        es_gt_volume = np.sum(gt_matrix)

        efs = (((ed_pred_volume - es_pred_volume)/ed_pred_volume) - ((ed_gt_volume - es_gt_volume)/ed_gt_volume))**2
    else:
        efs = 0

    return efs

if __name__ == '__main__':
    
    ## Metrics Unet pred&gt
    
    data_filepath = '/staff/ydli/projects/Med_Seg/seg/new_inf/kbr/v2_r0.9.2/fold5/hdf5_eval/'
    
    mode_list = ['ed', 'es']

    sid_list = []
    data_filelist = os.listdir(data_filepath)

    hausdorff_dis = []
    # chamfer_dis = []
    iou_3d = []
    dice_3d = []

    asd_list = []
    jaccard = []
    volume_sim = []
    ef_sim = []
    false_negative = []
    false_positive = []


    
    for file in os.listdir(data_filepath):
            sid = file.split('.')[-2]
            if 'ed' in sid and '3042' in sid :
                try:
                    print('3042 sid:', sid)
                    file_path = data_filepath+'/'+file
                    hdf5_file = h5py.File(file_path, 'r')
                    data_voxel = np.asarray(hdf5_file['data'])
                    preds_matrix = np.asarray(hdf5_file['inf_image'])
                    gt_matrix = np.asarray(hdf5_file['gt_label'])

                    hd = metric_recons_hd(preds_matrix, gt_matrix)

                    # cd = metric_recons_cd(pred_points, gt_points)

                    iou = metric_recons_IOU(preds_matrix, gt_matrix)

                    dice = metric_recons_DICE(preds_matrix, gt_matrix)
                
                    ed_pred_volume = np.sum(preds_matrix)
                    ed_gt_volume = np.sum(gt_matrix)
                    efs = get_sid_efs(sid ,ed_pred_volume, ed_gt_volume, data_filepath)

                    predict = copy.deepcopy(preds_matrix*1)
                    target = copy.deepcopy(gt_matrix)
                    predict = sitk.GetImageFromArray(predict)
                    target = sitk.GetImageFromArray(target)
                    predict = sitk.Cast(predict,sitk.sitkUInt8)
                    target = sitk.Cast(target,sitk.sitkUInt8)   
                    roi_pred = (torch.from_numpy(preds_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0) # 11HWD
                    roi_true = (torch.from_numpy(gt_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0)

                    result = cal_score(predict, target)
                    asd = cal_asd(roi_pred,roi_true)

                    print('metrics '+sid+' successfully!')
                except:
                    print('metrics '+sid+' failed!')
                else:
                    sid_list.append(sid)
                    hausdorff_dis.append(hd)
                    # chamfer_dis.append(cd)
                    iou_3d.append(iou)
                    dice_3d.append(dice)
                    asd_list.append(asd)
                    jaccard.append(result['Jaccard'])
                    volume_sim.append(abs(result['VolumeSimilarity']))
                    ef_sim.append(efs)
                    false_negative.append(result['FalseNegativeError'])
                    false_positive.append(result['FalsePositiveError'])
            else:
                try:
                    file_path = data_filepath+'/'+file
                    hdf5_file = h5py.File(file_path, 'r')
                    data_voxel = np.asarray(hdf5_file['data'])
                    preds_matrix = np.asarray(hdf5_file['inf_image'])
                    gt_matrix = np.asarray(hdf5_file['gt_label'])

                    hd = metric_recons_hd(preds_matrix, gt_matrix)

                    # cd = metric_recons_cd(pred_points, gt_points)

                    iou = metric_recons_IOU(preds_matrix, gt_matrix)

                    dice = metric_recons_DICE(preds_matrix, gt_matrix)

                    efs = 0

                    predict = copy.deepcopy(preds_matrix*1)
                    target = copy.deepcopy(gt_matrix)
                    predict = sitk.GetImageFromArray(predict)
                    target = sitk.GetImageFromArray(target)
                    predict = sitk.Cast(predict,sitk.sitkUInt8)
                    target = sitk.Cast(target,sitk.sitkUInt8)   
                    roi_pred = (torch.from_numpy(preds_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0) # 11HWD
                    roi_true = (torch.from_numpy(gt_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0)

                    result = cal_score(predict, target)
                    asd = cal_asd(roi_pred,roi_true)

                    print('metrics '+sid+' successfully!')
                except:
                    print('metrics '+sid+' failed!')
                else:
                    sid_list.append(sid)
                    hausdorff_dis.append(hd)
                    # chamfer_dis.append(cd)
                    iou_3d.append(iou)
                    dice_3d.append(dice)
                    asd_list.append(asd)
                    jaccard.append(result['Jaccard'])
                    volume_sim.append(abs(result['VolumeSimilarity']))
                    ef_sim.append(efs)
                    false_negative.append(result['FalseNegativeError'])
                    false_positive.append(result['FalsePositiveError'])
    print(len(sid_list))
    print(len(hausdorff_dis))
    print(hausdorff_dis[0])
    # print(len(chamfer_dis))
    # print(chamfer_dis[0])
    print(len(iou_3d))
    print(iou_3d[0])
    print(len(dice_3d))
    print(dice_3d[0])
    print(len(jaccard))
    print(jaccard[0])
    print(len(volume_sim))
    print(volume_sim[0])
    print(len(false_negative))
    print(false_negative[0])
    print(len(false_positive))
    print(false_positive[0])
    print(len(asd_list))
    print(asd_list[0])
    print(len(ef_sim))
    print(ef_sim[0])

    csv_dict = {'sid_mode':sid_list,
                'hausdorff_dis':hausdorff_dis, 'iou_3d':iou_3d, 'dice_3d':dice_3d, 
                'jaccard':jaccard, 'VolumeSimilarity':volume_sim, 'FalseNegativeError':false_negative, 'FalsePositiveError':false_positive, 'asd':asd_list, 'ef_sim':ef_sim}
    df = pd.DataFrame(csv_dict)
    df.to_csv('/staff/ydli/projects/Med_Seg/seg/new_inf/kbr/metrics_eval_convex_fold5.csv')

    # for file in os.listdir(data_filepath):
    #         sid = file.split('.')[-2]
    #         if 'ed' in sid and '3042' in sid :
    #                 print('3042 sid:', sid)
    #                 file_path = data_filepath+'/'+file
    #                 hdf5_file = h5py.File(file_path, 'r')
    #                 data_voxel = np.asarray(hdf5_file['data'])
    #                 preds_matrix = np.asarray(hdf5_file['inf_image'])
    #                 gt_matrix = np.asarray(hdf5_file['gt_label'])

    #                 hd = metric_recons_hd(preds_matrix, gt_matrix)

    #                 # cd = metric_recons_cd(pred_points, gt_points)

    #                 iou = metric_recons_IOU(preds_matrix, gt_matrix)

    #                 dice = metric_recons_DICE(preds_matrix, gt_matrix)
                
    #                 ed_pred_volume = np.sum(preds_matrix)
    #                 ed_gt_volume = np.sum(gt_matrix)
    #                 efs = get_sid_efs(sid ,ed_pred_volume, ed_gt_volume, data_filepath)
    #                 print(efs)

    #                 predict = copy.deepcopy(preds_matrix*1)
    #                 target = copy.deepcopy(gt_matrix)
    #                 predict = sitk.GetImageFromArray(predict)
    #                 target = sitk.GetImageFromArray(target)
    #                 predict = sitk.Cast(predict,sitk.sitkUInt8)
    #                 target = sitk.Cast(target,sitk.sitkUInt8)   
    #                 roi_pred = (torch.from_numpy(preds_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0) # 11HWD
    #                 roi_true = (torch.from_numpy(gt_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0)

    #                 result = cal_score(predict, target)
    #                 asd = cal_asd(roi_pred,roi_true)

    #                 print('metrics '+sid+' successfully!')
    #                 break
    #         else:
    #                 file_path = data_filepath+'/'+file
    #                 hdf5_file = h5py.File(file_path, 'r')
    #                 data_voxel = np.asarray(hdf5_file['data'])
    #                 preds_matrix = np.asarray(hdf5_file['inf_image'])
    #                 gt_matrix = np.asarray(hdf5_file['gt_label'])

    #                 hd = metric_recons_hd(preds_matrix, gt_matrix)

    #                 # cd = metric_recons_cd(pred_points, gt_points)

    #                 iou = metric_recons_IOU(preds_matrix, gt_matrix)

    #                 dice = metric_recons_DICE(preds_matrix, gt_matrix)

    #                 efs = 0

    #                 predict = copy.deepcopy(preds_matrix*1)
    #                 target = copy.deepcopy(gt_matrix)
    #                 predict = sitk.GetImageFromArray(predict)
    #                 target = sitk.GetImageFromArray(target)
    #                 predict = sitk.Cast(predict,sitk.sitkUInt8)
    #                 target = sitk.Cast(target,sitk.sitkUInt8)   
    #                 roi_pred = (torch.from_numpy(preds_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0) # 11HWD
    #                 roi_true = (torch.from_numpy(gt_matrix)==1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0)

    #                 result = cal_score(predict, target)
    #                 asd = cal_asd(roi_pred,roi_true)

    #                 print('metrics '+sid+' successfully!')
            
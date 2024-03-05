import numpy as np
import os
import pandas as pd
import pickle
import glob
import random
import h5py

import pyvista as pv
import open3d as o3d

from xml_reader import parseXml
import trimesh
from trimesh.voxel import creation
from trimesh import Trimesh
from Dataset.CSL import CSL

from scipy.spatial import ConvexHull

from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


def make_cardio_label_mesh2voxels(filename, mode):
    '''
    read cardio mesh ground truth from a xml file, mode record 'ed', 'es'
    then output a matrix which is the dense matrix voxels of this mesh (N*N*N) filled with True, False
    '''
    
    info_list, cali_info, mesh_text = parseXml(filename)

    verts, faces = read_kbr_mesh(mesh_text[mode])
    # print(verts[0])
    # print(verts.max())
    # print(verts.min())

    vector = np.mean(verts, axis=0)
    verts -= vector
    scale = 1.1 * np.max(np.absolute(verts))
    verts /= scale

    mesh = Trimesh(verts, faces)
    origin = np.zeros((3,))
    voxels = creation.local_voxelize(mesh, origin, pitch=0.016, radius=64, fill=True)
    # voxels = creation.local_voxelize(mesh, origin, pitch=0.008, radius=128, fill=True)
    # print(voxels)
    # voxels.show()
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)
    # print(matrix.shape)

    # output matrix with shape:(512, 512, 512) filled with True&False

    return matrix, scale, vector
    

def make_cardio_image_mesh2voxels(filename, scale, vector):
    '''
    read cardio plane mesh(train data) from ply,
    create a convex mesh for this mesh,
    then output a np.array (N, 3) which represents the voxels of this mesh
    '''

    # Load the .ply file
    mesh = trimesh.load_mesh(filename)
    moved_mesh = mesh.apply_translation(-vector)
    scaled_mesh = moved_mesh.apply_scale(1/scale)

    origin = np.zeros((3,))
    voxels = creation.local_voxelize(scaled_mesh, origin, pitch=0.016, radius=64, fill=True)
    # voxels = creation.local_voxelize(scaled_mesh, origin, pitch=0.008, radius=128, fill=True)
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)
    print(matrix.shape)

    # output matrix with shape:(512, 512, 512) filled with True&False

    return matrix

def make_cardio_profile_image_mesh2voxels(filename, scale, vector):
    '''
    read cardio plane mesh(train data) from ply,
    create a convex mesh for this mesh,
    then output a np.array (N, 3) which represents the voxels of this mesh
    '''

    # Load the .ply file
    mesh = trimesh.load_mesh(filename)
    moved_mesh = mesh.apply_translation(-vector)
    scaled_mesh = moved_mesh.apply_scale(1/scale)
    convex_hull = scaled_mesh.convex_hull

    origin = np.zeros((3,))
    voxels = creation.local_voxelize(convex_hull, origin, pitch=0.016, radius=64, fill=True)
    # voxels = creation.local_voxelize(scaled_mesh, origin, pitch=0.008, radius=128, fill=True)
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)
    # print(matrix.shape)


    shape = (129, 129, 129)
    voxel_array = np.zeros(shape)
    pitch = 0.016

    # print(type(scaled_mesh.vertices))
    for point in scaled_mesh.vertices:
        x, y, z = point

        voxel_x = int((x / pitch) + (shape[0] - 1) / 2)
        voxel_y = int((y / pitch) + (shape[1] - 1) / 2)
        voxel_z = int((z / pitch) + (shape[2] - 1) / 2)
        if 0 <= voxel_x < shape[0] and 0 <= voxel_y < shape[1] and 0 <= voxel_z < shape[2]:
            voxel_array[voxel_x, voxel_y, voxel_z] = 1

    voxel_array = np.delete(voxel_array, 0, 0)
    voxel_array = np.delete(voxel_array, 0, 1)
    voxel_array = np.delete(voxel_array, 0, 2)
    output_matrix = np.stack([matrix, voxel_array])

    return output_matrix

def make_cardio_profile_line_image_mesh2voxels(ply_filename, csl_filename, scale, vector):
    '''
    read cardio plane mesh(train data) from ply,
    create a convex mesh for this mesh,
    then output a np.array (N, 3) which represents the voxels of this mesh
    '''

    # Load the .ply file
    mesh = trimesh.load_mesh(ply_filename)
    moved_mesh = mesh.apply_translation(-vector)
    scaled_mesh = moved_mesh.apply_scale(1/scale)

    convex_hull = scaled_mesh.convex_hull

    origin = np.zeros((3,))
    voxels = creation.local_voxelize(convex_hull, origin, pitch=0.016, radius=64, fill=True)
    # voxels = creation.local_voxelize(scaled_mesh, origin, pitch=0.008, radius=128, fill=True)
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)
    # print(matrix.shape)


    shape = (129, 129, 129)
    voxel_array = np.zeros(shape)
    pitch = 0.016

    plane_verts_list = []

    csl =  CSL.from_csl_file(csl_filename)
    for plane in csl.planes:
        plane_verts = plane.vertices
        plane_verts -= vector
        plane_verts /= scale

        plane_verts_list.append(plane_verts)
    # print(type(scaled_mesh.vertices))
    for plane_verts in plane_verts_list:
        for i in range(plane_verts.shape[0]):
            p1 = plane_verts[i]
            p2 = plane_verts[(i + 1) % plane_verts.shape[0]]  # 下一个点，循环回到首点
            # print((i + 1) % plane_verts.shape[0])
            # print(plane_verts.shape[0])
            # print(plane_verts[0])
            
            direction_vector = p1 - p2
            length = np.linalg.norm(direction_vector)
            num_samples = int(length / pitch) + 1
            # print(num_samples)

            for i in range(num_samples):
                
                if num_samples == 1:
                    t = 0.0
                else:
                    t = i / (num_samples - 1)
                point = p1 + t * direction_vector
                voxel_index = ((point / pitch) + (np.array(shape) - 1) / 2).astype(int)
                # print(voxel_index)
                voxel_array[voxel_index[0], voxel_index[1], voxel_index[2]] = 1
                
    voxel_array = np.delete(voxel_array, 0, 0)
    voxel_array = np.delete(voxel_array, 0, 1)
    voxel_array = np.delete(voxel_array, 0, 2)
    # print(voxel_array.shape)
    count_true = np.sum(voxel_array)
    print('count_true:', count_true)
    output_matrix = np.stack([matrix, voxel_array])
    # output matrix with shape:(512, 512, 512) filled with True&False
    # print(output_matrix.shape)

    return output_matrix


def make_hdf5_for_Carver(gt_filepath, train_filepath, mode_list):

    sid_list = []
    train_matrixlist = []
    gt_matrixlist = []
    scale_list = []

    # print(os.listdir(gt_filepath))
    for filename in os.listdir(gt_filepath):
        
        xml_file = gt_filepath+'/'+filename
        sid = filename.split('_')[-1].split('.')[0]
        # print(sid) # sid = '10280'
        sid = 'SID' + '_' + filename.split('_')[-2] + '_' + sid   
        # print(sid) # sid = 'SID_3042_10280'

        for mode in mode_list: # mode 'es'&'ed'
            # make es&ed gt data .txt file
            sid_mode = sid + '_' + mode
            try:
                gt_matrix, scale, vector = make_kbr_label_mesh2voxels(xml_file, mode)
                print(sid_mode+' gt matrix created')

            except:
                print(' error: '+sid_mode+' gt matrix not created')
                print(sid_mode+' data created failed')

            else:
                try:
                    ply_id_file = train_filepath +'/*' +sid+'.ply'
                    csl_id_file = train_filepath +'/*' +sid+'.csl'
                    # len_file=len(glob.glob(id_file))
                    # print(len_file)
                    for ply_file in glob.glob(ply_id_file):
                        if mode in ply_file:
                            train_matrix = make_kbr_profile_image_mesh2voxels(ply_file, scale, vector) # make_kbr_profile_line_image_mesh2voxels(ply_file, csl_file, scale, vector)
                            print(sid_mode+' train matrix file created')
                except:
                    print(' error: '+sid_mode+' train matrix not created')
                    print(sid_mode+' data created failed')
                else:
                    sid_list.append(sid_mode)
                    scale_list.append(scale)
                    train_matrixlist.append(train_matrix)
                    gt_matrixlist.append(gt_matrix)
                    print(sid_mode+' data created successfully')
     
    unet_path = '...'
    if not os.path.exists(unet_path):
        os.makedirs(unet_path)

    for sid, image, label, scale in zip(sid_list, train_matrixlist, gt_matrixlist, scale_list):

        hdf5_path = unet_path+'/'+sid+'.hdf5'
        hdf5_file = h5py.File(hdf5_path, 'w')
        hdf5_file.create_dataset('image', data=image.astype(np.int16))
        hdf5_file.create_dataset('label', data=label.astype(np.uint8))
        hdf5_file.create_dataset('scale', data=scale.astype(np.float32))
        hdf5_file.close()
        print('create '+sid+' hdf5 file success!')
 
    # print(sid_list[0])
    # print(len(sid_list))


if __name__ == '__main__':

    gt_filepath = '..'
    train_filepath = '..'
    mode_list = ['es', 'ed']

    csv_filepath = '..'

    # make .hdf5 file for Carver from .ply&.xml(without ref)
    make_hdf5_for_Carver(gt_filepath, train_filepath, mode_list)

    



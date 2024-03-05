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

from kbr_mesh import KBRMesh
from kbr_slice import read_kbr_plane, read_kbr_mesh



def make_kbr_gt_mesh2voxels(filename, mode):
    '''
    read kbr mesh ground truth from a xml file, mode record 'ed', 'es'
    then output a matrix which is the dense matrix voxels of this mesh (N*N*N) filled with True, False
    '''
    
    info_list, cali_info, mesh_text = parseXml(filename)
    verts, faces = read_kbr_mesh(mesh_text[mode])
    verts -= np.mean(verts, axis=0)
    scale = 1.1
    verts /= scale * np.max(np.absolute(verts))
    # print(verts)
    mesh = Trimesh(verts, faces)
    voxels = creation.local_voxelize(mesh, mesh.centroid, pitch=0.005, radius=128, fill=True)
    # print(voxels)
    # voxels.show()
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)
    # print(matrix.shape)

    # output matrix with shape:(512, 512, 512) filled with True&False

    return matrix

def make_kbr_train_mesh2voxels(filename):
    '''
    read kbr plane mesh(train data) from ply,
    create a convex mesh for this mesh,
    then output a np.array (N, 3) which represents the voxels of this mesh
    '''
    
    # pcd = o3d.io.read_point_cloud(filename)
    # hull, _ = pcd.compute_convex_hull()
    # # print(type(hull))
    # # hull.scale(1 / np.max(hull.get_max_bound() - hull.get_min_bound()),
    # #        center=hull.get_center())
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(hull,
    #                                                           voxel_size=0.005)
    # voxels = voxel_grid.get_voxels()
    # indices = np.stack(list(vx.grid_index for vx in voxels)) # 
    # # print(indices)
    # # o3d.visualization.draw_geometries([voxel_grid])

    pcd = o3d.io.read_point_cloud(filename)
    pcd.estimate_normals()

    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd,
            1)

    # print(np.asarray(mesh.vertices))

    tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))
    
    convex_hull = trimesh.convex.convex_hull(tri_mesh, qhull_options='QbB Pp Qt', repair=True)
    voxels = creation.local_voxelize(convex_hull, convex_hull.centroid, pitch=0.005, radius=128, fill=True)
    # voxels.show()
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    # print(matrix.shape)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)

    # output matrix with shape:(512, 512, 512) filled with True&False

    return matrix



def make_kbr_label_mesh2voxels(filename, mode):
    '''
    read kbr mesh ground truth from a xml file, mode record 'ed', 'es'
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
    

def make_kbr_image_mesh2voxels(filename, scale, vector):
    '''
    read kbr plane mesh(train data) from ply,
    create a convex mesh for this mesh,
    then output a np.array (N, 3) which represents the voxels of this mesh
    '''

    # Load the .ply file
    mesh = trimesh.load_mesh(filename)

    # mesh.show()
    # # Create the convex hull
    # convex_hull = mesh.convex_hull
    # # print(np.mean(np.asarray(convex_hull.vertices), axis=0))
    # # print(np.asarray(convex_hull.vertices).max())
    # # print(np.asarray(convex_hull.vertices).min())
    # moved_mesh = convex_hull.apply_translation(-vector)
    # # print(np.mean(np.asarray(moved_mesh.vertices), axis=0))
    # # print(np.asarray(moved_mesh.vertices).max())
    # # print(np.asarray(moved_mesh.vertices).min())
    # scaled_mesh = moved_mesh.apply_scale(1/scale)
    # # print(np.mean(np.asarray(scaled_mesh.vertices), axis=0))
    # # print(np.asarray(scaled_mesh.vertices).max())
    # # print(np.asarray(scaled_mesh.vertices).min())
    # # print(type(convex_hull))

    # Create the mesh
    # print(np.mean(np.asarray(convex_hull.vertices), axis=0))
    # print(np.asarray(convex_hull.vertices).max())
    # print(np.asarray(convex_hull.vertices).min())
    moved_mesh = mesh.apply_translation(-vector)
    # print(np.mean(np.asarray(moved_mesh.vertices), axis=0))
    # print(np.asarray(moved_mesh.vertices).max())
    # print(np.asarray(moved_mesh.vertices).min())
    scaled_mesh = moved_mesh.apply_scale(1/scale)
    # print(np.mean(np.asarray(scaled_mesh.vertices), axis=0))
    # print(np.asarray(scaled_mesh.vertices).max())
    # print(np.asarray(scaled_mesh.vertices).min())
    # print(type(convex_hull))

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

def make_kbr_profile_image_mesh2voxels(filename, scale, vector):
    '''
    read kbr plane mesh(train data) from ply,
    create a convex mesh for this mesh,
    then output a np.array (N, 3) which represents the voxels of this mesh
    '''

    # Load the .ply file
    mesh = trimesh.load_mesh(filename)

    # mesh.show()
    # # Create the convex hull
    # convex_hull = mesh.convex_hull
    # # print(np.mean(np.asarray(convex_hull.vertices), axis=0))
    # # print(np.asarray(convex_hull.vertices).max())
    # # print(np.asarray(convex_hull.vertices).min())
    # moved_mesh = convex_hull.apply_translation(-vector)
    # # print(np.mean(np.asarray(moved_mesh.vertices), axis=0))
    # # print(np.asarray(moved_mesh.vertices).max())
    # # print(np.asarray(moved_mesh.vertices).min())
    # scaled_mesh = moved_mesh.apply_scale(1/scale)
    # # print(np.mean(np.asarray(scaled_mesh.vertices), axis=0))
    # # print(np.asarray(scaled_mesh.vertices).max())
    # # print(np.asarray(scaled_mesh.vertices).min())
    # # print(type(convex_hull))

    # Create the mesh
    # print(np.mean(np.asarray(convex_hull.vertices), axis=0))
    # print(np.asarray(convex_hull.vertices).max())
    # print(np.asarray(convex_hull.vertices).min())
    moved_mesh = mesh.apply_translation(-vector)
    # print(np.mean(np.asarray(moved_mesh.vertices), axis=0))
    # print(np.asarray(moved_mesh.vertices).max())
    # print(np.asarray(moved_mesh.vertices).min())
    scaled_mesh = moved_mesh.apply_scale(1/scale)
    # print(np.mean(np.asarray(scaled_mesh.vertices), axis=0))
    # print(np.asarray(scaled_mesh.vertices).max())
    # print(np.asarray(scaled_mesh.vertices).min())
    # print(type(convex_hull))
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
    # print(voxel_array.shape)
    # count_true = np.sum(voxel_array)
    # print('count_true:', count_true)
    output_matrix = np.stack([matrix, voxel_array])
    # output matrix with shape:(512, 512, 512) filled with True&False
    # print(output_matrix.shape)

    return output_matrix

def make_kbr_profile_line_image_mesh2voxels(ply_filename, csl_filename, scale, vector):
    '''
    read kbr plane mesh(train data) from ply,
    create a convex mesh for this mesh,
    then output a np.array (N, 3) which represents the voxels of this mesh
    '''

    # Load the .ply file
    mesh = trimesh.load_mesh(ply_filename)

    # mesh.show()
    # # Create the convex hull
    # convex_hull = mesh.convex_hull
    # # print(np.mean(np.asarray(convex_hull.vertices), axis=0))
    # # print(np.asarray(convex_hull.vertices).max())
    # # print(np.asarray(convex_hull.vertices).min())
    # moved_mesh = convex_hull.apply_translation(-vector)
    # # print(np.mean(np.asarray(moved_mesh.vertices), axis=0))
    # # print(np.asarray(moved_mesh.vertices).max())
    # # print(np.asarray(moved_mesh.vertices).min())
    # scaled_mesh = moved_mesh.apply_scale(1/scale)
    # # print(np.mean(np.asarray(scaled_mesh.vertices), axis=0))
    # # print(np.asarray(scaled_mesh.vertices).max())
    # # print(np.asarray(scaled_mesh.vertices).min())
    # # print(type(convex_hull))

    # Create the mesh
    # print(np.mean(np.asarray(convex_hull.vertices), axis=0))
    # print(np.asarray(convex_hull.vertices).max())
    # print(np.asarray(convex_hull.vertices).min())
    moved_mesh = mesh.apply_translation(-vector)
    # print(np.mean(np.asarray(moved_mesh.vertices), axis=0))
    # print(np.asarray(moved_mesh.vertices).max())
    # print(np.asarray(moved_mesh.vertices).min())
    scaled_mesh = moved_mesh.apply_scale(1/scale)
    # print(np.mean(np.asarray(scaled_mesh.vertices), axis=0))
    # print(np.asarray(scaled_mesh.vertices).max())
    # print(np.asarray(scaled_mesh.vertices).min())
    # print(type(convex_hull))
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
            
            # print(voxel_x_min, voxel_y_min, voxel_z_min)
            # print(voxel_x_max, voxel_y_max, voxel_z_max)

        # if 0 <= voxel_x_min < shape[0] and 0 <= voxel_x_max < shape[1] and 0 <= voxel_y_min < shape[1] and 0 <= voxel_y_max < shape[1] and 0 <= voxel_z_min < shape[1] and 0 <= voxel_z_max < shape[2]:
        #     voxel_array[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1
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

def make_kbr_poly_image_mesh2voxels(csl_filename, scale, vector):
    '''
    read kbr plane mesh(train data) from ply,
    create a convex mesh for this mesh,
    then output a np.array (N, 3) which represents the voxels of this mesh
    '''

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
            
            # print(voxel_x_min, voxel_y_min, voxel_z_min)
            # print(voxel_x_max, voxel_y_max, voxel_z_max)

        # if 0 <= voxel_x_min < shape[0] and 0 <= voxel_x_max < shape[1] and 0 <= voxel_y_min < shape[1] and 0 <= voxel_y_max < shape[1] and 0 <= voxel_z_min < shape[1] and 0 <= voxel_z_max < shape[2]:
        #     voxel_array[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = 1
    voxel_array = np.delete(voxel_array, 0, 0)
    voxel_array = np.delete(voxel_array, 0, 1)
    voxel_array = np.delete(voxel_array, 0, 2)
    # print(voxel_array.shape)
    count_true = np.sum(voxel_array)
    print('count_true:', count_true)

    # output matrix with shape:(512, 512, 512) filled with True&False
    # print(output_matrix.shape)

    return voxel_array

def make_points_voxels(points, scale, vector):

    mesh = trimesh.Trimesh(vertices=points)
    # print(mesh.vertices.shape)
    # print(mesh.vertices[0])
    # print(mesh.vertices.max())
    # print(mesh.vertices.min())
    # print(mesh)
    convex_hull = mesh.convex_hull
    # print(np.mean(np.asarray(convex_hull.vertices), axis=0))
    # print(np.asarray(convex_hull.vertices).max())
    # print(np.asarray(convex_hull.vertices).min())
    moved_mesh = convex_hull.apply_translation(-vector)
    # print(np.mean(np.asarray(moved_mesh.vertices), axis=0))
    # print(np.asarray(moved_mesh.vertices).max())
    # print(np.asarray(moved_mesh.vertices).min())
    scaled_mesh = moved_mesh.apply_scale(1/scale)
    # print(np.mean(np.asarray(scaled_mesh.vertices), axis=0))
    # print(np.asarray(scaled_mesh.vertices).max())
    # print(np.asarray(scaled_mesh.vertices).min())
    # print(type(scaled_mesh))
    # print(scaled_mesh.vertices[0])

    origin = np.zeros((3,))
    voxels = creation.local_voxelize(scaled_mesh, origin, pitch=0.016, radius=64, fill=True)
    # print(voxels)
    matrix = voxels.matrix
    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)
    matrix = np.delete(matrix, 0, 2)

    # print(matrix.shape)

    # output matrix with shape:(512, 512, 512) filled with True&False

    return matrix

def make_gt_pickle_txt_file(dump_filepath, xml_file, mode):

    txt_file = open(dump_filepath, 'wb')
    matrix = make_kbr_gt_mesh2voxels(xml_file, mode)
    pickle.dump(matrix, txt_file)
    txt_file.close()

    return

def make_train_pickle_txt_file(dump_filepath, ply_file):

    txt_file = open(dump_filepath, 'wb')
    matrix = make_kbr_train_mesh2voxels(ply_file)
    pickle.dump(matrix, txt_file)
    txt_file.close()

    return


def make_txt_file_for_UNet(gt_filepath, train_filepath, mode_list, data_path):

    # create data path
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("UNet Folder created")
    else:
        print("UNet Folder already exists")
    
    # create es&ed data path
    for mode in mode_list:
        mode_data_path = data_path + '/' + mode + '_data'
        if not os.path.exists(mode_data_path):
            os.makedirs(mode_data_path)
            print('UNet '+mode+ '_data Folder created')
        else:
            print('UNet '+mode+ '_data Folder already exists')

    # print(len(os.listdir(gt_filepath)))
    for filename in os.listdir(gt_filepath):

        xml_file = gt_filepath+'/'+filename

        # filename is like 'VpStudy_SID_3042_10280.xml'
        sid = filename.split('_')[-1].split('.')[0]
        # print(sid) # sid = '10280'
        sid = 'SID' + '_' + filename.split('_')[-2] + '_' + sid   
        # print(sid) # sid = 'SID_3042_10280'

        for mode in mode_list: # mode 'es'&'ed'
            mode_data_path = data_path + '/' + mode + '_data'
            sid_data_path = mode_data_path + '/' + sid
            if not os.path.exists(sid_data_path):
                os.makedirs(sid_data_path)
                print(sid+'_'+mode+' Folder created')
            else:
                print(sid+'_'+mode+' Folder already exists')

            sid_gt_data_path = mode_data_path + '/' + sid + '/' + 'gt'
            if not os.path.exists(sid_gt_data_path):
                os.makedirs(sid_gt_data_path)
                print(sid+'_'+mode+' gt Folder created')
            else:
                print(sid+'_'+mode+' gt Folder already exists')

            sid_train_data_path = mode_data_path + '/' + sid + '/' + 'train'
            if not os.path.exists(sid_train_data_path):
                os.makedirs(sid_train_data_path)
                print(sid+'_'+mode+' train Folder created')
            else:
                print(sid+'_'+mode+' train Folder already exists')

            # make es&ed gt data .txt file
            try:

                dump_filepath = sid_gt_data_path+ '/' + sid+'_'+mode+'_gt_data.txt'
                make_gt_pickle_txt_file(dump_filepath, xml_file, mode)
                print(sid+'_'+mode+' gt txt file created')
            except:
                print(' error: '+sid+'_'+mode+' gt txt file not created')


            id_file = train_filepath +'/*' +sid+'.ply'
            # len_file=len(glob.glob(id_file))
            # print(len_file)
            for file in glob.glob(id_file):
                if mode in file:
                    ply_file = file
                    try:
                        if len(os.listdir(sid_gt_data_path)) != 0:
                            dump_filepath = sid_train_data_path+ '/' + sid+'_'+mode+'_train_data.txt'
                            make_train_pickle_txt_file(dump_filepath, ply_file)
                            print(sid+'_'+mode+' train txt file created')
                            
                        else:
                            print(' error: '+sid+'_'+mode+' gt txt file does not exist')
                    except:
                        print(' error: '+sid+'_'+mode+' train txt file not created')
    
    return



def make_csv_file_for_UNet(mode_list, data_path):

    # make the .csv file
    sid_list = []
    train_filelist = []
    gt_filelist = []
    split_list = []
    ratio = 0.25

    for mode in mode_list:
        mode_data_path = data_path + '/' + mode + '_data'
        for sid in os.listdir(mode_data_path):
            # print(sid)
            train_path = mode_data_path + '/' + sid + '/' + 'train'
            gt_path = mode_data_path + '/' + sid + '/' + 'gt'

            if len(os.listdir(train_path)) != 0:

                sid_mode = sid + '_' + mode
                sid_list.append(sid_mode)

                train_filepath = train_path + '/' + sid+'_'+mode+'_train_data.txt'
                train_filelist.append(train_filepath)

                gt_filepath = gt_path + '/' + sid+'_'+mode+'_gt_data.txt'
                gt_filelist.append(gt_filepath)

                if random.uniform(0, 1) > ratio :

                    split_list.append('training')
                else:
                    split_list.append('validation')

    # print(sid_list[0])
    print(len(sid_list))
    # print(train_filelist[0])
    # print(len(train_filelist))
    # print(gt_filelist[0])
    # print(len(gt_filelist))
    # print(split_list)
    csv_dict = {'data_name':sid_list, 'train_filepath':train_filelist, 'gt_filepath':gt_filelist, 'split':split_list}
    df = pd.DataFrame(csv_dict)
    
    # save dataframe as .csv file
    df.to_csv('/staff/ydli/projects/OReX/Data/UNet/UNet_data.csv')

def read_pickle_saveas_hdf5(mode_list, data_path):

    # make the .csv file
    # sid_list = []
    # train_matrixlist = []
    # gt_matrixlist = []
    # split_list = []
    # ratio = 0.25
    unet_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5'

    for mode in mode_list:
        sid_list = []
        train_filelist = []
        gt_filelist = []
        mode_data_path = data_path + '/' + mode + '_data'
        for sid in os.listdir(mode_data_path):
            # print(sid)
            train_path = mode_data_path + '/' + sid + '/' + 'train'
            gt_path = mode_data_path + '/' + sid + '/' + 'gt'

            if len(os.listdir(train_path)) != 0:

                sid_mode = sid + '_' + mode
                sid_list.append(sid_mode)

                train_filepath = train_path + '/' + sid+'_'+mode+'_train_data.txt'
                train_filelist.append(train_filepath)

                gt_filepath = gt_path + '/' + sid+'_'+mode+'_gt_data.txt'
                gt_filelist.append(gt_filepath)

                # if random.uniform(0, 1) > ratio :

                #     split_list.append('training')
                # else:
                #     split_list.append('validation')
        # print(sid_list[0])
        length = len(sid_list)
        # print(train_filelist)
        for index in range(length):
            # print(index)
            sid  = sid_list[index]
            train_file = train_filelist[index]
            # print(train_file)
            gt_file = gt_filelist[index]

            train_pickle_file = open(train_file,'rb')
            image = pickle.load(train_pickle_file)
            train_pickle_file.close()

            gt_pickle_file = open(gt_file,'rb')
            label = pickle.load(gt_pickle_file)
            gt_pickle_file.close()

            hdf5_path = unet_path+'/'+sid+'.hdf5'
            hdf5_file = h5py.File(hdf5_path, 'w')
            hdf5_file.create_dataset('image', data=image.astype(np.int16))
            hdf5_file.create_dataset('label', data=label.astype(np .uint8))
            hdf5_file.close()
            print('create '+sid+' hdf5 file success!')
 
    # print(sid_list[0])
    # print(len(sid_list))


def make_hdf5_for_UNet(gt_filepath, train_filepath, mode_list):

    sid_list = []
    train_matrixlist = []
    gt_matrixlist = []
    scale_list = []

    # print(os.listdir(gt_filepath))
    for filename in os.listdir(gt_filepath):
        
        xml_file = gt_filepath+'/'+filename
        # print(filename)
        # filename is like 'VpStudy_SID_3042_10280.xml'
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
     
    unet_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_remove_8_128_backup'
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

def make_points_hdf5_for_UNet(gt_filepath, csv_filepath, mode_list):

    sid_list = []
    sample_list = []
    p_list = []
    p_labels_list = []
    train_matrixlist = []
    gt_matrixlist = []
    scale_list = []
    vector_list = []

    data_df = pd.read_csv(csv_filepath)
    info_list = data_df.to_dict('records')

    for info_dict in info_list:

        with open(info_dict['data_path'], 'rb') as f:
            sample = pickle.load(f) # sample['pc_points'], sample['pc_labels']
            sample_list.append(sample)

    # print(sample_sid_list[0])
    # print(len(sample_sid_list))
    # print(sample_phase_list[0])
    # print(len(sample_phase_list))
    # print(sample_list[0])
    # print(len(sample_list))

    # print(os.listdir(gt_filepath))
    for filename in os.listdir(gt_filepath):
        
        xml_file = gt_filepath+'/'+filename
        # print(filename)
        # filename is like 'VpStudy_SID_3042_10280.xml'
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
                    for sample in sample_list:
                        sid_phase = sample['sid'] + '_'+sample['phase']
                        # print(sid_phase)
                        if sid_phase == sid_mode:
                            points, p_labels = sample['pc_points'], sample['pc_labels']
                            train_matrix = make_points_voxels(points, scale, vector)
                            print(sid_mode+' train matrix created')

                except:
                    print(' error: '+sid_mode+' train matrix not created')
                    print(sid_mode+' data created failed')
                else:
                    sid_list.append(sid_mode)
                    train_matrixlist.append(train_matrix)
                    gt_matrixlist.append(gt_matrix)
                    p_list.append(points)
                    p_labels_list.append(p_labels)
                    vector_list.append(vector)
                    scale_list.append(scale)
                    print(sid_mode+' data created successfully')

     
    unet_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_points_128_backup'
    if not os.path.exists(unet_path):
        os.makedirs(unet_path)

    for sid, image, label, points, p_label, vector, scale in zip(sid_list, train_matrixlist, gt_matrixlist, p_list, p_labels_list, vector_list, scale_list):

        hdf5_path = unet_path+'/'+sid+'.hdf5'
        hdf5_file = h5py.File(hdf5_path, 'w')
        hdf5_file.create_dataset('image', data=image.astype(np.int16))
        hdf5_file.create_dataset('label', data=label.astype(np.uint8))
        hdf5_file.create_dataset('points', data=points.astype(np.uint8))
        hdf5_file.create_dataset('p_label', data=p_label.astype(np.uint8))
        hdf5_file.create_dataset('vector', data=vector.astype(np.uint8))
        hdf5_file.create_dataset('scale', data=scale.astype(np.uint8))
        hdf5_file.close()
        print('create '+sid+' hdf5 file success!')
 
    # print(sid_list[0])
    # print(len(sid_list))



if __name__ == '__main__':
    
    # ## Test count the number of voxels in our voxel data

    # gt_filename = '/staff/ydli/projects/OReX/Data/kbr_patient_backup/VpStudy_SID_3039_16440.xml' # 
    # train_filename = '/staff/ydli/projects/OReX/Data/kbr_without_ref_backup/kbr_ed_heart_SID_3039_16440.ply' # '/staff/ydli/projects/OReX/Data/kbr_patient_backup/VpStudy_SID_3042_10280.xml'
    
    
    # gt_voxel, scale, vector = make_kbr_label_mesh2voxels(gt_filename, 'es')
    # print(gt_voxel.shape)
    # count_true = np.sum(gt_voxel) / 134217728
    # print(count_true)
    # train_voxel = make_kbr_image_mesh2voxels(train_filename, scale, vector)
    # print(train_voxel.shape)
    # count_true = np.sum(gt_voxel) / 134217728
    # print(count_true)
    # print(scale)
    # print(vector)

    # gt_verts, gt_faces, _, _ = measure.marching_cubes(gt_voxel, level=0)
    # train_verts, train_faces, _, _ = measure.marching_cubes(train_voxel, level=0)

    # fig = plt.figure(figsize=(10, 7))

    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter(train_verts[:, 0], train_verts[:, 1], train_verts[:, 2], color="blue", s=10)
    # ax.plot_trisurf(train_verts[:, 0], train_verts[:, 1], train_faces, train_verts[:, 2], cmap='Spectral', lw=1)

    # ax = fig.add_subplot(122, projection='3d')
    # ax.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], color="blue", s=10)
    # ax.plot_trisurf(gt_verts[:, 0], gt_verts[:, 1], gt_faces, gt_verts[:, 2], cmap='Spectral', lw=1)

    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_zlabel("Z-axis")

    # plt.show()
    # # plt.savefig('/staff/ydli/projects/OReX/Data/UNet/png/figure.png')

    # ## Test glob

    # train_filepath = '/staff/ydli/projects/OReX/Data/kbr_backup'
    # sid = 'SID_3042_10280'

    # id_file = train_filepath +'/*' +sid+'.ply'
    # # len_file=len(glob.glob(id_file))
    # # print(len_file)
    # for file in glob.glob(id_file):
    #     print(file)




    ## Run this line to make the .csv file which record the .txt file(voxel file)'s path

    gt_filepath = '/staff/ydli/projects/OReX/Data/kbr_patient_backup'
    train_filepath = '/staff/ydli/projects/OReX/Data/kbr_wo_ref_remove_8'
    mode_list = ['es', 'ed']

    csv_filepath = '/staff/wangzhaohui/codes/flownet3d_pytorch/instance/kbr_mesh_pc/info_list.csv'

    # # ## Test failed file 3039_10770_es

    # for filename in os.listdir(gt_filepath):
        
    #     xml_file = gt_filepath+'/'+filename
    #     # print(filename)
    #     # filename is like 'VpStudy_SID_3042_10280.xml'
    #     sid = filename.split('_')[-1].split('.')[0]
    #     # print(sid) # sid = '10280'
    #     sid = 'SID' + '_' + filename.split('_')[-2] + '_' + sid   
    #     # print(sid) # sid = 'SID_3042_10280'

    #     for mode in mode_list: # mode 'es'&'ed'
    #         # make es&ed gt data .txt file
    #         sid_mode = sid + '_' + mode
            
    #         gt_matrix, scale, vector = make_kbr_label_mesh2voxels(xml_file, mode)
    #         print(sid_mode+' gt matrix created')

    #         ply_id_file = train_filepath +'/*' +sid+'.ply'
    #         csl_id_file = train_filepath +'/*' +sid+'.csl'
    #         # len_file=len(glob.glob(id_file))
    #         # print(len_file)
    #         for ply_file in glob.glob(ply_id_file):
    #             if mode in ply_file:
    #                 for csl_file in glob.glob(csl_id_file):
    #                     if mode in csl_file:
    #                         train_matrix = make_kbr_profile_line_image_mesh2voxels(ply_file, csl_file, scale, vector) # make_kbr_profile_line_image_mesh2voxels(ply_file, csl_file, scale, vector)
    #                         print(sid_mode+' train matrix file created')

    #     unet_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_line_128_backup'
    #     if not os.path.exists(unet_path):
    #         os.makedirs(unet_path)


    #     hdf5_path = unet_path+'/'+sid+'.hdf5'
    #     hdf5_file = h5py.File(hdf5_path, 'w')
    #     hdf5_file.create_dataset('image', data=train_matrix.astype(np.int16))
    #     hdf5_file.create_dataset('label', data=gt_matrix.astype(np.uint8))
    #     hdf5_file.create_dataset('scale', data=scale.astype(np.float32))
    #     hdf5_file.close()
    #     print('create '+sid+' hdf5 file success!')

    #     break

    # # make the .txt file
    # make_txt_file_for_UNet(gt_filepath, train_filepath, mode_list, data_path)


    # # make the .csv file
    # # print(len(os.listdir(gt_filepath)))
    # # print(os.listdir(gt_filepath))
    # read_pickle_saveas_hdf5(mode_list, data_path)


    # make .hdf5 file for UNet from .ply&.xml(without ref)
    make_hdf5_for_UNet(gt_filepath, train_filepath, mode_list)

    # make .hdf5 file for UNet from .ply&.xml(without ref)
    # make_points_hdf5_for_UNet(gt_filepath, csv_filepath, mode_list)



    # #  Test

    # sid_list = []
    # sample_list = []
    # sample_sid_list = []
    # sample_phase_list = []
    # p_list = []
    # p_labels_list = []
    # train_matrixlist = []
    # gt_matrixlist = []
    # scale_list = []
    # vector_list = []

    # data_df = pd.read_csv(csv_filepath)
    # info_list = data_df.to_dict('records')

    # for info_dict in info_list:

    #     with open(info_dict['data_path'], 'rb') as f:
    #         sample = pickle.load(f) # sample['pc_points'], sample['pc_labels']
    #         sample_list.append(sample)
    #     sample_sid_list.append(info_dict['sid'])
    #     sample_phase_list.append(info_dict['phase'])

    # # print(len(sample_sid_list))
    # # print(len(sample_phase_list))
    # # print(len(sample_list))
    # # print(sample_phase_list[0], sample_phase_list[1])

    # for filename in os.listdir(gt_filepath):
        
    #     xml_file = gt_filepath+'/'+filename
    #     # print(filename)
    #     # filename is like 'VpStudy_SID_3042_10280.xml'
    #     sid = filename.split('_')[-1].split('.')[0]
    #     # print(sid) # sid = '10280'
    #     sid = 'SID' + '_' + filename.split('_')[-2] + '_' + sid   
    #     # print(sid) # sid = 'SID_3042_10280'

    #     for mode in mode_list: # mode 'es'&'ed'
    #         # make es&ed gt data .txt file
    #         sid_mode = sid + '_' + mode

    #         gt_matrix, scale, vector = make_kbr_label_mesh2voxels(xml_file, mode)
    #         print(sid_mode+' gt matrix created')

    #         for sample in sample_list:
    #             sid_phase = sample['sid'] + '_'+sample['phase']
    #             # print(sid_phase)
    #             if sid_phase == sid_mode:
    #                 # print(sid_phase)
    #                 points, p_labels = sample['pc_points'], sample['pc_labels']
    #                 print(points[0])
    #                 # print(p_labels.shape)
    #                 train_matrix = make_points_voxels(points, scale, vector)
    #                 print(sid_phase+' train matrix created')

    #                 sid_list.append(sid_mode)
    #                 train_matrixlist.append(train_matrix)
    #                 gt_matrixlist.append(gt_matrix)
    #                 p_list.append(points)
    #                 p_labels_list.append(p_labels)
    #                 vector_list.append(vector)
    #                 scale_list.append(scale)
    #                 print(sid_phase+' data created successfully')
        



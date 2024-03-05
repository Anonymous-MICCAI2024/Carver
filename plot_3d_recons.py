import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import h5py
import open3d as o3d
import os

import trimesh



if __name__ == "__main__":

    # Test inf voxel Data

    # data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_profile_mesh_128_backup'
    data_path = '/staff/ydli/projects/Med_Seg/seg/new_inf/kbr/v3.12.2/fold2/hdf5_eval'
    png_path = '/staff/ydli/projects/Med_Seg/seg/new_inf/kbr/v1.4.0/fold3/input_png/'
    if not os.path.exists(png_path):
                        os.makedirs(png_path)
    # # Create a sample voxel grid (replace this with your actual voxel data)
                        
    # for file in os.listdir(data_path):
    #     hdf5_filepath = data_path+'/'+file
    #     hdf5_file = h5py.File(hdf5_filepath, 'r')
    #     data_voxel = np.asarray(hdf5_file['data'])
    #     inf_voxel = np.asarray(hdf5_file['inf_image'])
    #     gt_voxel = np.asarray(hdf5_file['gt_label'])

    #     sid = os.path.basename(hdf5_filepath).split('.')[-2]
    #     print('sid:', sid)
    #     print(gt_voxel.shape)

    #     # Use marching cubes to create a mesh
    #     data_verts, data_faces, _, _ = measure.marching_cubes(data_voxel, level=0)
    #     inf_verts, inf_faces, _, _ = measure.marching_cubes(inf_voxel, level=0)
    #     gt_verts, gt_faces, _, _ = measure.marching_cubes(gt_voxel, level=0)

    #     # Plot the resulting mesh
    #     fig = plt.figure(figsize=(10, 7))

    #     ax = fig.add_subplot(131, projection='3d')
    #     ax.scatter(data_verts[:, 0], data_verts[:, 1], data_verts[:, 2], color="blue", s=10)
    #     ax.plot_trisurf(data_verts[:, 0], data_verts[:, 1], data_faces, data_verts[:, 2], cmap='Spectral', lw=1)

    #     ax = fig.add_subplot(132, projection='3d')
    #     ax.scatter(inf_verts[:, 0], inf_verts[:, 1], inf_verts[:, 2], color="blue", s=10)
    #     ax.plot_trisurf(inf_verts[:, 0], inf_verts[:, 1], inf_faces, inf_verts[:, 2], cmap='Spectral', lw=1)

    #     ax = fig.add_subplot(133, projection='3d')
    #     ax.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], color="blue", s=10)
    #     ax.plot_trisurf(gt_verts[:, 0], gt_verts[:, 1], gt_faces, gt_verts[:, 2], cmap='Spectral', lw=1)

    #     ax.set_xlabel("X-axis")
    #     ax.set_ylabel("Y-axis")
    #     ax.set_zlabel("Z-axis")

    #     # plt.show()
    #     plt.savefig(png_path+sid+'.png')

    #     print('save '+sid+' image .png success')
    
    # # Test eval hdf5
    # file = 'SID_3042_10280_ed.hdf5'
    # hdf5_filepath = data_path+'/'+file
    # hdf5_file = h5py.File(hdf5_filepath, 'r')
    # data_voxel = np.asarray(hdf5_file['data'])
    # inf_voxel = np.asarray(hdf5_file['inf_image'])
    # gt_voxel = np.asarray(hdf5_file['gt_label'])

    # sid = os.path.basename(hdf5_filepath).split('.')[-2]
    # print('sid:', sid)
    # print(gt_voxel.shape)

    # # fig = plt.figure()
    # # ax = fig.gca(projection='3d')
    # # ax.set_aspect('equal')

    # # ax.voxels(gt_voxel, edgecolor="k")

    # # plt.show()

    # # Use marching cubes to create a mesh
    # data_verts, data_faces, _, _ = measure.marching_cubes(data_voxel, level=0)
    # inf_verts, inf_faces, _, _ = measure.marching_cubes(inf_voxel, level=0)
    # gt_verts, gt_faces, _, _ = measure.marching_cubes(gt_voxel, level=0)

    # # Plot the resulting mesh
    # fig = plt.figure(figsize=(10, 7))

    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(data_verts[:, 0], data_verts[:, 1], data_verts[:, 2], color="blue", s=10)
    # # ax.plot_trisurf(data_verts[:, 0], data_verts[:, 1], data_faces, data_verts[:, 2], cmap='Spectral', lw=1)

    # ax = fig.add_subplot(121, projection='3d')
    # ax.scatter(inf_verts[:, 0], inf_verts[:, 1], inf_verts[:, 2], color="blue", s=10)
    # ax.plot_trisurf(inf_verts[:, 0], inf_verts[:, 1], inf_faces, inf_verts[:, 2], cmap='Spectral', lw=1)

    # ax = fig.add_subplot(122, projection='3d')
    # ax.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], color="blue", s=10)
    # ax.plot_trisurf(gt_verts[:, 0], gt_verts[:, 1], gt_faces, gt_verts[:, 2], cmap='Spectral', lw=1)

    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_zlabel("Z-axis")

    # # plt.show()
    # plt.savefig(png_path+sid+'.png')

    # print('save '+sid+' image .png success')


    # # Test made voxel Data

    # data_path = '/staff/ydli/projects/OReX/Data/UNet/hdf5_128_backup'
    hdf5_filepath = data_path+'/SID_3042_10160_ed.hdf5'
    hdf5_file = h5py.File(hdf5_filepath, 'r')
    data_voxel = np.asarray(hdf5_file['inf_image'])
    print(data_voxel.shape)
    # profile_voxel = data_voxel[1]
    # print(profile_voxel.shape)
    # data_voxel = data_voxel[0]
    # print(data_voxel.shape)
    gt_voxel = np.asarray(hdf5_file['gt_label'])

    sid = os.path.basename(hdf5_filepath).split('.')[-2]
    print('sid:', sid)
    print(gt_voxel.shape)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制体素网格1
    # x1, y1, z1 = data_voxel.nonzero()
    # ax.scatter(x1, y1, z1, c='r', marker='s', alpha=0.005)

    # # 绘制体素网格2
    # x2, y2, z2 = profile_voxel.nonzero()
    # ax.scatter(x2, y2, z2, c='b', marker='s', alpha=1.0)

    # # 设置坐标轴范围
    # ax.set_xlim([0, data_voxel.shape[0]])
    # ax.set_ylim([0, data_voxel.shape[1]])
    # ax.set_zlim([0, data_voxel.shape[2]])
    orex_mesh = trimesh.load_mesh('/staff/ydli/projects/OReX/output/directory/kbr/kbr_ed_heart_SID_3042_10160/mesh_last_300.obj')

    # 获取顶点和面数据
    orex_verts = orex_mesh.vertices
    orex_faces = orex_mesh.faces

    inf_verts, inf_faces, _, _ = measure.marching_cubes(data_voxel, level=0)
    gt_verts, gt_faces, _, _ = measure.marching_cubes(gt_voxel, level=0)

    # Plot the resulting mesh
    fig = plt.figure(figsize=(10, 7))

    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(data_verts[:, 0], data_verts[:, 1], data_verts[:, 2], color="blue", s=10)
    # ax.plot_trisurf(data_verts[:, 0], data_verts[:, 1], data_faces, data_verts[:, 2], cmap='Spectral', lw=1)

    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(gt_verts[:, 0], gt_verts[:, 1], gt_verts[:, 2], color="blue", s=10)
    ax.plot_trisurf(gt_verts[:, 0], gt_verts[:, 1], gt_faces, gt_verts[:, 2], cmap='Spectral', lw=1)
    ax.set_title('gt')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.tick_params(axis='z', which='both', left=False, right=False, labelleft=False)

    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(inf_verts[:, 0], inf_verts[:, 1], inf_verts[:, 2], color="blue", s=10)
    ax.plot_trisurf(inf_verts[:, 0], inf_verts[:, 1], inf_faces, inf_verts[:, 2], cmap='Spectral', lw=1)
    ax.set_title('pred')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.tick_params(axis='z', which='both', left=False, right=False, labelleft=False)


    ax = fig.add_subplot(133, projection='3d')
    ax.scatter(orex_verts[:, 0], orex_verts[:, 1], orex_verts[:, 2], color="blue", s=10)
    ax.plot_trisurf(orex_verts[:, 0], orex_verts[:, 1], orex_faces, orex_verts[:, 2], cmap='Spectral', lw=1)
    ax.set_title('orex')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    ax.tick_params(axis='z', which='both', left=False, right=False, labelleft=False)


    

    # plt.show()
    plt.savefig('/staff/ydli/projects/OReX/Data/UNet/png_128/'+sid+'_pred_orex_line.png')

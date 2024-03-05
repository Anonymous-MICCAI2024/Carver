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

   
    x_axis_data = [0,1,2,3,4,5,6,7,8]

    y_dual_data1 = [98.25, 98.00, 97.71, 97.12, 96.34, 96.18, 95.80, 95.51, 95.34]
    y_point_data2 = [97.91, 97.67, 97.37, 96.97, 96.10, 94.25, 93.32, 91.96, 87.27]
    y_convex_data3 = [96.33, 96.11, 95.61, 95.16, 94.29, 94.23, 93.76, 93.64, 93.55]
    y_orex_data4 = [95.17, 94.22, 92.76, 90.55, 86.51, 84.79, 82.46, 82.21, 79.84]

            
    #画图 
    plt.plot(x_axis_data, y_dual_data1, 'm*--', alpha=0.5, linewidth=1, label='Our method')#'
    plt.plot(x_axis_data, y_point_data2, 'bs--', alpha=0.5, linewidth=1, label='Single-channel Point sample')
    plt.plot(x_axis_data, y_convex_data3, 'go--', alpha=0.5, linewidth=1, label='Single-channel convex hull sample')
    plt.plot(x_axis_data, y_orex_data4, 'c^--', alpha=0.5, linewidth=1, label='OReX')

    
    plt.legend()  #显示上面的label
    plt.xlabel('Remove Plane number')
    plt.ylabel('Dice')#accuracy
    
    #plt.ylim(-1,1)#仅设置y轴坐标范围
    # plt.show()
    plt.savefig('/staff/ydli/projects/OReX/Data/UNet/png_128/'+'remove_plane.png')
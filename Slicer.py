import argparse
import os
import random
import shutil
import pyvista as pv
import numpy as np
import trimesh
from matplotlib.path import Path
from meshcut import cross_section
from shapely.geometry import LinearRing
from stl import mesh as mesh2
from tqdm import tqdm

from Dataset.CSL import CSL, ConnectedComponent, Plane
from Dataset.Helpers import plane_origin_from_params
from kbr_slice import read_kbr_mesh, read_kbr_plane
from kbr_plane import make_plane_transform
from xml_reader import parseXml


def _get_verts_faces(filename):
    scene = trimesh.load_mesh(filename)

    verts = scene.vertices
    faces = scene.faces

    verts -= np.mean(verts, axis=0)
    scale = 1.1
    verts /= scale * np.max(np.absolute(verts))

    return verts, faces


def make_csl_from_mesh(filename, save_path, n_slices):
    verts, faces, = _get_verts_faces(filename)
    model_name = os.path.split(filename)[-1].split('.')[0]

    plane_normals, ds = _get_random_planes(n_slices)

    plane_normals = (plane_normals.T / np.linalg.norm(plane_normals.T, axis=0)).T
    plane_origins = [plane_origin_from_params((*n, d)) for n, d in zip(plane_normals, ds)]

    ccs_per_plane = [cross_section(verts, faces, plane_orig=o, plane_normal=n) for o, n in tqdm(list(zip(plane_origins, plane_normals)))]

    csl = _csl_from_mesh(model_name, plane_origins, plane_normals, ds, ccs_per_plane)

    _save_sliced_mesh(csl, faces, model_name, save_path, verts)
    return csl

def make_csl_from_kbr(file_path, save_path, mode):
    info_list, cali_info, mesh_text = parseXml(file_path)
    # print(info_list[1]['sid'])
    verts, faces, = read_kbr_mesh(mesh_text[mode])
    plane_vector = np.mean(verts, axis=0)
    verts -= plane_vector
    # print(plane_vector)
    plane_scale = 1.1
    plane_scale *= np.max(np.absolute(verts))
    # # print(plane_scale)
    verts /= plane_scale
    # print(verts)
    model_name = 'kbr_'+mode+'_heart_' + info_list[0]['sid']

    plane_normals, ds, trans_matrixs_list = _get_kbr_planes(info_list, cali_info, plane_scale, plane_vector, mode)
    plane_origins = [plane_origin_from_params((*n, d)) for n, d in zip(plane_normals, ds)]


    # poly_faces = [[3, a, b, c ] for a, b, c in faces]
    # mesh_poly = pv.PolyData(verts, poly_faces)
    # pl = pv.Plotter()
    # pl.add_mesh(mesh_poly, show_edges=True, color= 'red')
    # pl.add_axes(line_width=5)
    # for normal, origin in zip(plane_normals, plane_origins):
    #     pl.add_mesh(pv.Plane(origin, normal))
    # pl.show()

    ccs_per_plane = [cross_section(verts, faces, plane_orig=o, plane_normal=n) for o, n in tqdm(list(zip(plane_origins, plane_normals)))]
    
    csl = _csl_from_mesh(model_name, plane_origins, plane_normals, ds, ccs_per_plane)

    _save_sliced_mesh(csl, faces, model_name, save_path, verts)
    return csl

def make_csl_from_kbr_without_ref(file_path, save_path, mode):
    info_list, cali_info, mesh_text = parseXml(file_path)
    # print(info_list[1]['sid'])
    verts, faces, = read_kbr_mesh(mesh_text[mode])
    plane_vector = np.zeros((3, ))
    # verts -= plane_vector
    # print(plane_vector)
    plane_scale = 1
    # plane_scale *= np.max(np.absolute(verts))
    # # # print(plane_scale)
    # verts /= plane_scale
    # print(verts.max())
    # print(verts.min())
    model_name = 'kbr_'+mode+'_heart_' + info_list[0]['sid']

    plane_normals, ds, trans_matrixs_list= _get_kbr_planes(info_list, cali_info, plane_scale, plane_vector, mode)
    # print(plane_normals)
    # print(ds)
    plane_origins = [plane_origin_from_params((*n, d)) for n, d in zip(plane_normals, ds)]


    # poly_faces = [[3, a, b, c ] for a, b, c in faces]
    # mesh_poly = pv.PolyData(verts, poly_faces)
    # pl = pv.Plotter()
    # pl.add_mesh(mesh_poly, show_edges=True, color= 'red')
    # pl.add_axes(line_width=5, box=True)
    # for normal, origin in zip(plane_normals, plane_origins):
    #     pl.add_mesh(pv.Plane(origin, normal, i_size=500, j_size=500))
    # pl.show()
    
    ccs_per_plane = [cross_section(verts, faces, plane_orig=o, plane_normal=n) for o, n in tqdm(list(zip(plane_origins, plane_normals)))]
    # print('ccs_per_plane len:', len(ccs_per_plane))
    # print('ccs type:', type(ccs_per_plane[0][0]))
    # print('ccs 0:', ccs_per_plane[0][0])
    print('ccs 0 shape:', ccs_per_plane[0][0].shape)
    csl = _csl_from_mesh(model_name, plane_origins, plane_normals, ds, ccs_per_plane)

    _save_sliced_mesh(csl, faces, model_name, save_path, verts)
    return csl

def make_csl_from_kbr_without_ref_remove_m(file_path, save_path, mode, remove_num):
    info_list, cali_info, mesh_text = parseXml(file_path)
    # print(info_list[1]['sid'])
    verts, faces, = read_kbr_mesh(mesh_text[mode])
    plane_vector = np.zeros((3, ))
    # verts -= plane_vector
    # print(plane_vector)
    plane_scale = 1
    # plane_scale *= np.max(np.absolute(verts))
    # # # print(plane_scale)
    # verts /= plane_scale
    # print(verts.max())
    # print(verts.min())
    model_name = 'kbr_'+mode+'_heart_' + info_list[0]['sid']

    plane_normals, ds, trans_matrixs_list= _get_kbr_planes_remove_m(info_list, cali_info, plane_scale, plane_vector, mode, remove_num)
    # print(plane_normals)
    # print(ds)
    plane_origins = [plane_origin_from_params((*n, d)) for n, d in zip(plane_normals, ds)]


    # poly_faces = [[3, a, b, c ] for a, b, c in faces]
    # mesh_poly = pv.PolyData(verts, poly_faces)
    # pl = pv.Plotter()
    # pl.add_mesh(mesh_poly, show_edges=True, color= 'red')
    # pl.add_axes(line_width=5, box=True)
    # for normal, origin in zip(plane_normals, plane_origins):
    #     pl.add_mesh(pv.Plane(origin, normal, i_size=500, j_size=500))
    # pl.show()
    
    ccs_per_plane = [cross_section(verts, faces, plane_orig=o, plane_normal=n) for o, n in tqdm(list(zip(plane_origins, plane_normals)))]
    # print('ccs_per_plane len:', len(ccs_per_plane))
    # print('ccs type:', type(ccs_per_plane[0][0]))
    # print('ccs 0:', ccs_per_plane[0][0])
    # print('ccs 0 shape:', ccs_per_plane[0][0].shape)
    csl = _csl_from_mesh(model_name, plane_origins, plane_normals, ds, ccs_per_plane)

    _save_sliced_mesh(csl, faces, model_name, save_path, verts)
    return csl

def make_csl_from_kbr_with_ref_remove_m(file_path, save_path, mode, remove_num):
    info_list, cali_info, mesh_text = parseXml(file_path)
    # print(info_list[1]['sid'])
    verts, faces, = read_kbr_mesh(mesh_text[mode])
    plane_vector = np.mean(verts, axis=0)
    verts -= plane_vector
    # print(plane_vector)
    plane_scale = 1.1
    plane_scale *= np.max(np.absolute(verts))
    # # # print(plane_scale)
    verts /= plane_scale
    # print(verts.max())
    # print(verts.min())
    model_name = 'kbr_'+mode+'_heart_' + info_list[0]['sid']

    plane_normals, ds, trans_matrixs_list= _get_kbr_planes_remove_m(info_list, cali_info, plane_scale, plane_vector, mode, remove_num)
    # print(plane_normals)
    # print(ds)
    plane_origins = [plane_origin_from_params((*n, d)) for n, d in zip(plane_normals, ds)]


    # poly_faces = [[3, a, b, c ] for a, b, c in faces]
    # mesh_poly = pv.PolyData(verts, poly_faces)
    # pl = pv.Plotter()
    # pl.add_mesh(mesh_poly, show_edges=True, color= 'red')
    # pl.add_axes(line_width=5, box=True)
    # for normal, origin in zip(plane_normals, plane_origins):
    #     pl.add_mesh(pv.Plane(origin, normal, i_size=500, j_size=500))
    # pl.show()
    
    ccs_per_plane = [cross_section(verts, faces, plane_orig=o, plane_normal=n) for o, n in tqdm(list(zip(plane_origins, plane_normals)))]
    # print('ccs_per_plane len:', len(ccs_per_plane))
    # print('ccs type:', type(ccs_per_plane[0][0]))
    # print('ccs 0:', ccs_per_plane[0][0])
    # print('ccs 0 shape:', ccs_per_plane[0][0].shape)
    csl = _csl_from_mesh(model_name, plane_origins, plane_normals, ds, ccs_per_plane)

    _save_sliced_mesh(csl, faces, model_name, save_path, verts)
    return csl

def _save_sliced_mesh(csl, faces, model_name, save_path, verts):

    my_mesh = mesh2.Mesh(np.zeros(len(faces), dtype=mesh2.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            my_mesh.vectors[i][j] = verts[f[j], :]

    # print(my_mesh)
    my_mesh.save(os.path.join(save_path, f'{model_name}.stl'))
    csl.to_ply(os.path.join(save_path, f'{model_name}.ply'))
    csl.to_file(os.path.join(save_path, f'{model_name}.csl'))

def _get_random_planes(n_slices):
    plane_normals = np.random.randn(n_slices, 3)

    ds = -1 * (np.random.random_sample(n_slices) * 2 - 1)
    return plane_normals, ds

def _get_kbr_planes(info_list, cali_info, plane_scale, plane_vector, mode):
    # return plane's normals and ds

    trans_matrixs_list = []
    normal_list = []
    d_list = []

    # t_matrix = np.zeros((4,4))
    # t_matrix[[0,1,2,3], [2,1,0,3]] = 1
    # print(t_matrix)

    i_matrix = np.zeros((4, 3))
    # print(i_matrix)
    plane_vector = np.insert(plane_vector, 3, 0)
    plane_vector = plane_vector.T
    # print(plane_vector)
    i_matrix = np.insert(i_matrix, 3, plane_vector, axis=1)
    # print('i_matrix')
    # print(i_matrix)

    s_matrix = np.eye(4)
    s_matrix[:3,:3] = s_matrix[:3,:3] / plane_scale

    for plane in info_list:
        depth_label = plane['depth_label']
        enablePatientMovementCorrection = 0
        patient_sensor_orientation = np.array(plane[mode+'_patient_orientation'])[[3,0,1,2]]
        patient_initial_orientation = np.array(plane['init_patient_orientation'])[[3,0,1,2]]
        sensor_orientation = np.array(plane[mode+'_sensor_orientation'])[[3,0,1,2]]
        patient_sensor_position = np.array(plane[mode+'_patient_location'])
        patient_initial_position = np.array(plane['init_patient_location'])
        sensor_posiotion = np.array(plane[mode+'_sensor_location'])
        c_orien_matrix = np.reshape(cali_info['calibration_rotation_matrix'], (3,3))
        c_pos_vector = np.array(cali_info['calibration_translation_vector'])
        for depth in cali_info['depth_list']:
            if depth['depth_label'] == depth_label:
                depth_XMillimetersPerPixel = depth['x_millmeter_per_pixel']
                depth_YMillimetersPerPixel = depth['y_millmeter_per_pixel']
                depth_OriginPixel_X = depth['origin_x']
                depth_OriginPixel_Y = depth['origin_y']
        # print(plane_scale)
        trans_matrix = make_plane_transform(enablePatientMovementCorrection, patient_sensor_orientation, patient_initial_orientation, sensor_orientation, patient_sensor_position, patient_initial_position, sensor_posiotion, c_orien_matrix, c_pos_vector, depth_XMillimetersPerPixel, depth_YMillimetersPerPixel, depth_OriginPixel_X, depth_OriginPixel_Y)
        # trans_matrix = np.dot(t_matrix, trans_matrix)
        trans_matrix -= i_matrix
        trans_matrix = np.dot(s_matrix, trans_matrix)
        trans_matrixs_list.append(trans_matrix)

    for matrix in trans_matrixs_list:
        normal, d = read_kbr_plane(matrix)
        # d /= plane_scale
        normal_list.append(normal)
        d_list.append(d)

    normals = np.array(normal_list)
    ds = np.array(d_list)
    return normals, ds, trans_matrixs_list

def _get_kbr_planes_remove_m(info_list, cali_info, plane_scale, plane_vector, mode, remove_num):
    # return plane's normals and ds

    trans_matrixs_list = []
    normal_list = []
    d_list = []

    # t_matrix = np.zeros((4,4))
    # t_matrix[[0,1,2,3], [2,1,0,3]] = 1
    # print(t_matrix)

    i_matrix = np.zeros((4, 3))
    # print(i_matrix)
    plane_vector = np.insert(plane_vector, 3, 0)
    plane_vector = plane_vector.T
    # print(plane_vector)
    i_matrix = np.insert(i_matrix, 3, plane_vector, axis=1)
    # print('i_matrix')
    # print(i_matrix)

    s_matrix = np.eye(4)
    s_matrix[:3,:3] = s_matrix[:3,:3] / plane_scale

    plane_num = len(info_list)
    print('plane_num: ', plane_num)

    if (plane_num - remove_num)>4 :
        remove_ids = random.sample(range(plane_num), remove_num)
        print(remove_ids)
    else:
        remove_ids = random.sample(range(plane_num), plane_num-4)
        print(remove_ids)

    for i in range(len(info_list)):
        if i not in remove_ids:
            plane = info_list[i]

            depth_label = plane['depth_label']
            enablePatientMovementCorrection = 0
            patient_sensor_orientation = np.array(plane[mode+'_patient_orientation'])[[3,0,1,2]]
            patient_initial_orientation = np.array(plane['init_patient_orientation'])[[3,0,1,2]]
            sensor_orientation = np.array(plane[mode+'_sensor_orientation'])[[3,0,1,2]]
            patient_sensor_position = np.array(plane[mode+'_patient_location'])
            patient_initial_position = np.array(plane['init_patient_location'])
            sensor_posiotion = np.array(plane[mode+'_sensor_location'])
            c_orien_matrix = np.reshape(cali_info['calibration_rotation_matrix'], (3,3))
            c_pos_vector = np.array(cali_info['calibration_translation_vector'])
            for depth in cali_info['depth_list']:
                if depth['depth_label'] == depth_label:
                    depth_XMillimetersPerPixel = depth['x_millmeter_per_pixel']
                    depth_YMillimetersPerPixel = depth['y_millmeter_per_pixel']
                    depth_OriginPixel_X = depth['origin_x']
                    depth_OriginPixel_Y = depth['origin_y']
            # print(plane_scale)
            trans_matrix = make_plane_transform(enablePatientMovementCorrection, patient_sensor_orientation, patient_initial_orientation, sensor_orientation, patient_sensor_position, patient_initial_position, sensor_posiotion, c_orien_matrix, c_pos_vector, depth_XMillimetersPerPixel, depth_YMillimetersPerPixel, depth_OriginPixel_X, depth_OriginPixel_Y)
            # trans_matrix = np.dot(t_matrix, trans_matrix)
            trans_matrix -= i_matrix
            trans_matrix = np.dot(s_matrix, trans_matrix)
            trans_matrixs_list.append(trans_matrix)

    for matrix in trans_matrixs_list:
        normal, d = read_kbr_plane(matrix)
        # d /= plane_scale
        normal_list.append(normal)
        d_list.append(d)

    normals = np.array(normal_list)
    ds = np.array(d_list)
    return normals, ds, trans_matrixs_list

def _plane_from_mesh(ccs, plane_params, normal, origin, plane_id, csl):
    connected_components = []
    vertices = np.empty(shape=(0, 3))

    to_plane_cords = _get_to_plane_cords(ccs[0][0], normal, origin)

    for cc in ccs:
        # this does not handle non-empty holes
        if len(cc) > 2:
            is_hole, parent_cc_idx = _is_cc_hole(cc, ccs, to_plane_cords)

            oriented_cc = _orient_polyline(cc, is_hole, to_plane_cords)

            vert_start = len(vertices)
            if is_hole:
                connected_components.append(
                    ConnectedComponent(parent_cc_idx, 2, list(range(vert_start, vert_start + len(cc)))))
            else:
                connected_components.append(ConnectedComponent(-1, 1, list(range(vert_start, vert_start + len(cc)))))
            vertices = np.concatenate((vertices, oriented_cc))

    return Plane(plane_id, plane_params, vertices, connected_components, csl)


def _csl_from_mesh(model_name, plane_origins, plane_normals, ds, ccs_per_plane):
    def plane_gen(csl):
        planes = []
        i = 1

        for origin, normal, d, ccs in zip(plane_origins, plane_normals, ds, ccs_per_plane):
            plane_params = (*normal, d)

            if len(ccs) > 0:
                planes.append(_plane_from_mesh(ccs, plane_params, normal, origin, i, csl))
                i += 1
        return planes

    return CSL(model_name, plane_gen, n_labels=2)


def _get_to_plane_cords(point_on_plane, normal, origin):
    b0 = point_on_plane - origin
    b0 /= np.linalg.norm(b0)
    b1 = np.cross(normal, b0)
    transformation_matrix = np.array([b0, b1])

    def to_plane_cords(xyzs):
        alinged = xyzs - origin
        return np.array([transformation_matrix @ v for v in alinged])

    return to_plane_cords


def _is_cc_hole(cc, ccs, transform):
    is_hole = False
    parent_cc_idx = None

    point_inside_cc = transform(cc[0:1])
    for i, other_cc in enumerate(ccs):
        if other_cc is cc:
            continue
        shape_vertices = list(transform(other_cc)) + [[0, 0]]
        shape_codes = [Path.MOVETO] + [Path.LINETO] * (len(other_cc) - 1) + [Path.CLOSEPOLY]
        path = Path(shape_vertices, shape_codes)
        if path.contains_points(point_inside_cc)[0]:
            # todo not necessarily 1 but enough for my purposes
            is_hole = True
            parent_cc_idx = i
            break
    return is_hole, parent_cc_idx


def _orient_polyline(verts, is_hole, to_plane_cords):
    if is_hole == LinearRing(to_plane_cords(verts)).is_ccw:
        oriented_verts = verts[::-1]
    else:
        oriented_verts = verts
    return oriented_verts


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='slice a mesh')
    # parser.add_argument('input', type=str, help='path to mesh')
    # parser.add_argument('out_dir', type=str, help='out directory to save outputs')
    
    # args = parser.parse_args()

    # input_path = '/staff/ydli/projects/OReX/VpStudy_bak.xml'
    # out_path = '/staff/ydli/projects/OReX/Data/kbr'
    # print(f'Slicing kbr')
    # csl_ed = make_csl_from_ed_kbr(input_path, out_path)
    # csl_es = make_csl_from_es_kbr(input_path, out_path)

    # print(f'Generated csl={csl.model_name} non-empty slices={len([p for p in csl.planes if not p.is_empty])}, n edges={len(csl)} '
    #       f'Artifacts at: {args.out_dir}')

    # for filepath in os.listdir('/staff/ydli/projects/OReX/Data/xmls'):
    #     file_path = '/staff/ydli/projects/OReX/Data/xmls/'+filepath
    #     file_old_name = file_path + '/VpStudy.xml'
    #     file_name = file_path + '/VpStudy_' + filepath + '.xml'
    #     os.rename(file_old_name, file_name)

    # new_filepath = '/staff/ydli/projects/OReX/Data/kbr_patient_backup'
    # for filepath in os.listdir('/staff/ydli/projects/OReX/Data/xmls'):
    #      file_path = '/staff/ydli/projects/OReX/Data/xmls/'+filepath
    #      # old_filepath_list.append(filename)
    #      file_name = '/VpStudy_' + filepath + '.xml'
    #      # old_filename_list.append(filename)
    #      shutil.copy(file_path + file_name, new_filepath + file_name)
    
    # file_name = 'VpStudy_SID_3042_10280.xml'
    # input_path = '/staff/ydli/projects/OReX/Data/kbr_patient_backup/' + file_name
    # out_path = '/staff/ydli/projects/OReX/trash/'

    # ed_mode = 'ed'
    # csl_ed = make_csl_from_kbr(input_path, out_path, ed_mode)
    # es_mode = 'es'
    # csl_es = make_csl_from_kbr(input_path, out_path, es_mode)

    # for filename in os.listdir('/staff/ydli/projects/OReX/Data/kbr_patient_backup'):
    #     # print(filename)
    #     input_path = '/staff/ydli/projects/OReX/Data/kbr_patient_backup/' + filename
    #     out_path = '/staff/ydli/projects/OReX/Data/kbr_w/o_ref_test'
    #     # print(input_path)
    #     # break
    #     print(f'Slicing ' + filename)
    #     try:
    #         ed_mode = 'ed'
    #         csl_ed = make_csl_from_kbr_without_ref(input_path, out_path, ed_mode)
    #         es_mode = 'es'
    #         csl_es = make_csl_from_kbr_without_ref(input_path, out_path, es_mode)
    #     except:
    #         print('slice error')

    data_path = '/staff/ydli/projects/OReX/Data/kbr_patient_backup/'
    with_ref_out_path = '/staff/ydli/projects/OReX/Data/kbr_w_ref_remove_8'
    # without_ref_out_path = '/staff/ydli/projects/OReX/Data/kbr_wo_ref_remove_2'
    if not os.path.exists(with_ref_out_path):
        os.makedirs(with_ref_out_path)
    # if not os.path.exists(without_ref_out_path):
    #     os.makedirs(without_ref_out_path)

    for filename in os.listdir('/staff/ydli/projects/OReX/Data/kbr_patient_backup'):
        # print(filename)
        try:
            input_path = data_path+filename
            # print(input_path)
            print(f'Slicing ' + filename)
            remove_num = 8
            ed_mode = 'ed'
            # csl_ed_wo_ref = make_csl_from_kbr_without_ref_remove_m(input_path, without_ref_out_path, ed_mode, remove_num)
            csl_ed_w_ref = make_csl_from_kbr_with_ref_remove_m(input_path, with_ref_out_path, ed_mode, remove_num)
            print(f'Slicing ' + filename+' ed success')
            es_mode = 'es'
            # csl_es_wo_ref = make_csl_from_kbr_without_ref_remove_m(input_path, without_ref_out_path, es_mode, remove_num)
            csl_es_w_ref = make_csl_from_kbr_with_ref_remove_m(input_path, with_ref_out_path, es_mode, remove_num)
            print(f'Slicing ' + filename+' es success')
        except:
            print(f'Slicing ' + filename+' failed')


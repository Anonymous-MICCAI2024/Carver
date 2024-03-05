import numpy as np
from os import scandir
from xml.dom import minidom
from kbr_mesh import KBRMesh

STRUCTS = [
    'endo_point_surface',
    'rv_septum',
    'pulmonic_valve',
    'Sub_Pulmonary',
    'tricuspid_annulus',
    'basal_bulge_rv',
    'FourC_RX',
    'septal_edge',
    # 'spherepoint'
]

def fixed_writexml(self, writer, indent="", addindent="", newl=""):  
    # indent = current indentation  
    # addindent = indentation to add to higher levels  
    # newl = newline string  
    writer.write(indent+"<" + self.tagName)  

    attrs = self._get_attributes()  
    a_names = attrs.keys()  
    # a_names.sort()  

    for a_name in a_names:  
        writer.write(" %s=\"" % a_name)  
        minidom._write_data(writer, attrs[a_name].value)  
        writer.write("\"")  
    if self.childNodes:  
        if len(self.childNodes) == 1 and self.childNodes[0].nodeType == minidom.Node.TEXT_NODE:  
            writer.write(">")  
            self.childNodes[0].writexml(writer, "", "", "")  
            writer.write("</%s>%s" % (self.tagName, newl))  
            return  
        writer.write(">%s"%(newl))  
        for node in self.childNodes:  
            if node.nodeType is not minidom.Node.TEXT_NODE:  
                node.writexml(writer,indent+addindent,addindent,newl)  
        writer.write("%s</%s>%s" % (indent,self.tagName,newl))  
    else:  
        writer.write("/>%s"%(newl))  

minidom.Element.writexml = fixed_writexml  

def parsePoints(struct_nodes, scan_name):
    point_dict = {}
    for struct_name in STRUCTS:
        point_dict[struct_name] = []
    for struct_node in struct_nodes:
        struct_name = struct_node.getAttribute('Name')
        for point_node in struct_node.getElementsByTagName('Point'):
            # print(point_node.getAttribute('ScanName'), '-', scan_name)
            if point_node.getAttribute('ScanName') == scan_name:
                x = point_node.getElementsByTagName('X')[0].firstChild.nodeValue
                y = point_node.getElementsByTagName('Y')[0].firstChild.nodeValue
                # print(len(point_node.getElementsByTagName('X')))
                point_dict[struct_name].append([float(x), float(y)])
    return point_dict

def parseShape(root_node):
    cali_node = root_node.getElementsByTagName('Calibration')[0]
    width = cali_node.getElementsByTagName('FrameWidth')[0].firstChild.nodeValue
    height = cali_node.getElementsByTagName('FrameHeight')[0].firstChild.nodeValue
    return (int(height), int(width))


def parseFrameTrans(frame_node):
    # Sensor position and rotation
    sensor_node = frame_node.getElementsByTagName('SensorReadings')[0]
    sensor_location_node = sensor_node.getElementsByTagName('Location')[0]
    sensor_x = sensor_location_node.getElementsByTagName('X')[0].firstChild.nodeValue
    sensor_y = sensor_location_node.getElementsByTagName('Y')[0].firstChild.nodeValue
    sensor_z = sensor_location_node.getElementsByTagName('Z')[0].firstChild.nodeValue
    sensor_pos = (float(sensor_x), float(sensor_y), float(sensor_z))

    sensor_orien_node = sensor_node.getElementsByTagName('Orientation')[0]
    sensor_rx = sensor_orien_node.getElementsByTagName('X')[0].firstChild.nodeValue
    sensor_ry = sensor_orien_node.getElementsByTagName('Y')[0].firstChild.nodeValue
    sensor_rz = sensor_orien_node.getElementsByTagName('Z')[0].firstChild.nodeValue
    sensor_rw = sensor_orien_node.getElementsByTagName('W')[0].firstChild.nodeValue
    sensor_rotation = (float(sensor_rx), float(sensor_ry), float(sensor_rz), float(sensor_rw))
    
    # Patient sensor position and rotation
    
    patient_sensor_node = frame_node.getElementsByTagName('PatientSensorReadings')[0]
    patient_sensor_location_node = patient_sensor_node.getElementsByTagName('Location')[0]
    sensor_x = patient_sensor_location_node.getElementsByTagName('X')[0].firstChild.nodeValue
    sensor_y = patient_sensor_location_node.getElementsByTagName('Y')[0].firstChild.nodeValue
    sensor_z = patient_sensor_location_node.getElementsByTagName('Z')[0].firstChild.nodeValue
    patient_sensor_pos = (float(sensor_x), float(sensor_y), float(sensor_z))

    patient_sensor_orien_node = patient_sensor_node.getElementsByTagName('Orientation')[0]
    sensor_rx = patient_sensor_orien_node.getElementsByTagName('X')[0].firstChild.nodeValue
    sensor_ry = patient_sensor_orien_node.getElementsByTagName('Y')[0].firstChild.nodeValue
    sensor_rz = patient_sensor_orien_node.getElementsByTagName('Z')[0].firstChild.nodeValue
    sensor_rw = patient_sensor_orien_node.getElementsByTagName('W')[0].firstChild.nodeValue
    patient_sensor_rotation = (float(sensor_rx), float(sensor_ry), float(sensor_rz), float(sensor_rw))
    
    return  sensor_pos, sensor_rotation, patient_sensor_pos, patient_sensor_rotation


def parseInitPatientSensor(init_patient_node):
    
    patient_sensor_location_node = init_patient_node.getElementsByTagName('Location')[0]
    sensor_x = patient_sensor_location_node.getElementsByTagName('X')[0].firstChild.nodeValue
    sensor_y = patient_sensor_location_node.getElementsByTagName('Y')[0].firstChild.nodeValue
    sensor_z = patient_sensor_location_node.getElementsByTagName('Z')[0].firstChild.nodeValue
    init_patient_pos = (float(sensor_x), float(sensor_y), float(sensor_z))

    patient_sensor_orien_node = init_patient_node.getElementsByTagName('Orientation')[0]
    sensor_rx = patient_sensor_orien_node.getElementsByTagName('X')[0].firstChild.nodeValue
    sensor_ry = patient_sensor_orien_node.getElementsByTagName('Y')[0].firstChild.nodeValue
    sensor_rz = patient_sensor_orien_node.getElementsByTagName('Z')[0].firstChild.nodeValue
    sensor_rw = patient_sensor_orien_node.getElementsByTagName('W')[0].firstChild.nodeValue
    init_patient_orien = (float(sensor_rx), float(sensor_ry), float(sensor_rz), float(sensor_rw))
    
    return init_patient_pos, init_patient_orien


def parseCalibrationDepth(depth_node):
    depth_label = depth_node.getElementsByTagName('DepthLabel')[0].firstChild.nodeValue
    origin_point_node = depth_node.getElementsByTagName('OriginPixel')[0]
    origin_x = origin_point_node.getElementsByTagName('X')[0].firstChild.nodeValue
    origin_y = origin_point_node.getElementsByTagName('Y')[0].firstChild.nodeValue
    x_millmeter_per_pixel = depth_node.getElementsByTagName('XMillimetersPerPixel')[0].firstChild.nodeValue
    y_millmeter_per_pixel = depth_node.getElementsByTagName('YMillimetersPerPixel')[0].firstChild.nodeValue
    
    return {
        'depth_label': depth_label,
        'origin_x': float(origin_x),
        'origin_y': float(origin_y),
        'x_millmeter_per_pixel': float(x_millmeter_per_pixel),
        'y_millmeter_per_pixel': float(y_millmeter_per_pixel)
    }

def parseCalibrationTransform(calibration_node):
    rotate_matrix_node = calibration_node.getElementsByTagName('CalibrationRotationMatrix')[0]
    matrix_elements = []
    vector_elements = []
    for i in range(1, 4):
        for j in range(1, 4):
            elmt = rotate_matrix_node.getElementsByTagName(f'M{i}{j}')[0]
            matrix_elements.append(float(elmt.firstChild.nodeValue))
    translation_vector_node =  calibration_node.getElementsByTagName('CalibrationTranslationVector')[0]

    for tag_name in ('X', 'Y', 'Z'):
        elmt = translation_vector_node.getElementsByTagName(tag_name)[0]
        vector_elements.append(float(elmt.firstChild.nodeValue))

    return matrix_elements, vector_elements

def parseCalibration(calibration_node):
    depth_list_node = calibration_node.getElementsByTagName('UltrasoundDepthList')[0]
    depth_node_list = depth_list_node.getElementsByTagName('CalibrationDepth')
    
    depth_list = []
    for depth_node in depth_node_list:
        depth_list.append(parseCalibrationDepth(depth_node))
    
    calibration_rotmat, calibration_trasvec = parseCalibrationTransform(calibration_node)
    
    return {
        'depth_list': depth_list,
        'calibration_rotation_matrix': calibration_rotmat,
        'calibration_translation_vector': calibration_trasvec
    }


def parseMesh(root_node):
    analysis_node = root_node.getElementsByTagName('Analyses')[0].getElementsByTagName('Analysis')[0]
    es_mesh_node = analysis_node.getElementsByTagName('PhaseEs')[0].getElementsByTagName('Mesh')[0]
    ed_mesh_node = analysis_node.getElementsByTagName('PhaseEd')[0].getElementsByTagName('Mesh')[0]
    
    es_mesh_text = es_mesh_node.firstChild.nodeValue
    ed_mesh_text = ed_mesh_node.firstChild.nodeValue
    
    return {
        'es': es_mesh_text,
        'ed': ed_mesh_text
    }




def parseXml(xml_path):
    '''
    return a list of dict {
        'sid',
        'scan_name',
        'ed',
        'es',
        'ed_points',
        'es_points',
    }
    '''
    dom_tree = minidom.parse(xml_path)
    root_node = dom_tree.documentElement
    sid = root_node.getAttribute('StudyName').replace(':', '_').replace('.', '_')
    scan_list = []
    
    calibration_node = root_node.getElementsByTagName('Calibration')[0]
    calibration_info = parseCalibration(calibration_node)
    
    mesh_text = parseMesh(root_node)
    for scan_node in root_node.getElementsByTagName('StudyFrameIndices'):
        scan_dict = {}
        scan_dict['sid'] = sid
        scan_dict['scan_name'] = scan_node.getAttribute('ScanName')
        scan_dict['depth_label'] = scan_node.getAttribute('CalibrationDepthLabel')
        scan_dict['ed'] = int(scan_node.getAttribute('EDFrameIndex'))
        scan_dict['es'] = int(scan_node.getAttribute('ESFrameIndex'))
        
        ed_node = root_node.getElementsByTagName('PhaseEd')[0]
        es_node = root_node.getElementsByTagName('PhaseEs')[0]
        assert ed_node.parentNode.getAttribute('AnalysisType') == 'RV'
        assert es_node.parentNode.getAttribute('AnalysisType') == 'RV'
        ed_struct_nodes = ed_node.getElementsByTagName('AnatomicStructure')
        es_struct_nodes = es_node.getElementsByTagName('AnatomicStructure')
        scan_dict['ed_points'] = parsePoints(ed_struct_nodes, scan_dict['scan_name'])
        scan_dict['es_points'] = parsePoints(es_struct_nodes, scan_dict['scan_name'])
        
        frame_nodes = scan_node.parentNode.getElementsByTagName('Frame')
        
        es_frame_node = frame_nodes[scan_dict['es']]
        ed_frame_node = frame_nodes[scan_dict['ed']]
        sensor_pos, sensor_rotation, patient_sensor_pos, patient_sensor_rotation = parseFrameTrans(ed_frame_node)
        
        scan_dict['ed_sensor_location'] = sensor_pos
        scan_dict['ed_sensor_orientation'] = sensor_rotation
        scan_dict['ed_patient_location'] = patient_sensor_pos
        scan_dict['ed_patient_orientation'] = patient_sensor_rotation
        
        sensor_pos, sensor_rotation, patient_sensor_pos, patient_sensor_rotation = parseFrameTrans(es_frame_node)
        scan_dict['es_sensor_location'] = sensor_pos
        scan_dict['es_sensor_orientation'] = sensor_rotation
        scan_dict['es_patient_location'] = patient_sensor_pos
        scan_dict['es_patient_orientation'] = patient_sensor_rotation
        
        init_patient_node = scan_node.parentNode.getElementsByTagName('PatientInitialPosition')[0]
        init_patient_pos, init_patient_orien = parseInitPatientSensor(init_patient_node)
        scan_dict['init_patient_location'] = init_patient_pos
        scan_dict['init_patient_orientation'] = init_patient_orien
        
        # scan_dict['frame_shape'] = parseShape(root_node)
        scan_dict['height'], scan_dict['width'] = parseShape(root_node)
        scan_list.append(scan_dict)
    return scan_list, calibration_info, mesh_text

def setAttributesByDict(node, attr_dict):
    for key, value in attr_dict.items():
        node.setAttribute(key, value)

def removeChildren(node, children_name):
    for child in node.getElementsByTagName(children_name):
        node.removeChild(child)

class XmlReader():
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.dom_tree = minidom.parse(xml_path)
        self.root_node = self.dom_tree.documentElement

    def __find_scan_node(self, scan_name):
        for scan_node in self.root_node.getElementsByTagName('StudyFrameIndices'):
            if scan_node.getAttribute('ScanName') == scan_name:
                return scan_node
        return None

    def read_scan_info(self, scan_name):

        '''
        width, height
        '''
        info_dict = {}

        scan_node = self.__find_scan_node(scan_name)
        frame_node = scan_node.parentNode.getElementsByTagName('Frame')[0]
        info_dict['width'] = int(frame_node.getAttribute('Width'))
        info_dict['height'] = int(frame_node.getAttribute('Height'))
        
        return info_dict

class XmlWriter():
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.dom_tree = minidom.parse(xml_path)
        self.root_node = self.dom_tree.documentElement
        es_node = self.root_node.getElementsByTagName('PhaseEs')[0]
        ed_node = self.root_node.getElementsByTagName('PhaseEd')[0]
        self.es_struct_node = es_node.getElementsByTagName('AnatomicStructure')
        self.ed_struct_node = ed_node.getElementsByTagName('AnatomicStructure')
        
    
    def newPoint(self, x, y, scan_name, phase):
        point_attrib = {
            'Phase': phase,
            'm_scanIndex': '0',
            'ScanName': scan_name
        }
        point_node = self.dom_tree.createElement('Point')
        pos_node = self.dom_tree.createElement('ScreenPosition')
        x_node, y_node = self.dom_tree.createElement('X'), self.dom_tree.createElement('Y')
        
        x_node.appendChild(self.dom_tree.createTextNode(str(x)))
        y_node.appendChild(self.dom_tree.createTextNode(str(y)))
        setAttributesByDict(point_node, point_attrib)
        # print('x_node: ', x_node.toxml())
        # print('y_node: ', y_node.toxml())
        
        pos_node.appendChild(x_node)
        pos_node.appendChild(y_node)
        # print('pos_node: ', pos_node.toxml())
        point_node.appendChild(pos_node)
        # print('point_node: ', point_node.toxml())
        
        return point_node
        
    def clearPoints(self):
        for struct_node in self.es_struct_node:
            removeChildren(struct_node, 'Point')
        for struct_node in self.ed_struct_node:
            removeChildren(struct_node, 'Point')
    
    def addPointsForScan(self, points, scan_name, phase):
        # TODO: use dict to switch ED/ES
        if phase == 'ES':
            struct_node_list = self.es_struct_node
        elif phase == 'ED':
            struct_node_list = self.ed_struct_node
        for i, struct_node in enumerate(struct_node_list):
            cur_struct = struct_node.getAttribute('Name')
            for point in points[cur_struct]:
                point_node = self.newPoint(point[0], point[1], scan_name, phase)
                struct_node.appendChild(point_node)
                
    def write(self, save_path):
        with open(save_path, 'w', encoding='UTF-8') as save_file:
            self.dom_tree.writexml(save_file, addindent='  ',newl='\n', encoding='UTF-8')
            print('xml file saved!')
    

if __name__ == '__main__':

    path = '/staff/wangzhaohui/ultrasound/frame_detection/prepare_data/SID_3042_10280/VpStudy.xml'
    dicom_dir = ''

    # 1. inputs of one sample
    '''
    list of :
        seq_path
        ed, es
        sid, origin_sid
        scan_name
        labeled_points
    '''

    # 2. forward
    '''
    for scan in scan_list:
        seq = read_file(seq_path)
        input_frames = seq[ed,es]
        net = ...
        output_i = net(input_frames)
    '''

    # 3. postprocess
    '''
    for scan in scan_list:
        points_i = map2points(output_i)
    '''

    # 4. write xml file
    '''
    writer = XmlWriter('path')
    for scan in scan_list:
        writer.addPointsForScan(points_i[0], scan, 'ES')
        writer.addPointsForScan(points_i[1], scan, 'ED')
    '''

    file_path = '/staff/wangzhaohui/codes/kbr_recons/instance/xmls/VpStudy_bak.xml'
    info_list, cali_info, mesh_text = parseXml(file_path)
    print(info_list[0])
    print(np.reshape(cali_info['calibration_rotation_matrix'], (3,3)))
    print(mesh_text['es'])
    mesh = KBRMesh(mesh_text['ed'])
    # print(mesh.points.shape)
    mesh.to_poly().plot()
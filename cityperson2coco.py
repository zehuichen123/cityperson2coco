import numpy as np
import scipy.io as si
import os
import json
import time

## YOUR CONFIGURATION
cityperson_annotation_dir = '/home/tusimple/Documents/data/cityperson/annotations'
new_anntation_dir = '/home/tusimple/Documents/data/cityperson/annotations'
data_type = 'val'               # train / val
box_type = 'visible'            # visible / full

## DATASET CONFIGURATION
START_BOX_ID = 1
image_id = 1
bbox_id = START_BOX_ID
image = {}
annotation = {}
categories = {}

def load_annotation():
    file_name = 'anno_' + data_type + '.mat'
    anno_data = si.loadmat(os.path.join(cityperson_annotation_dir, file_name))
    return anno_data

def cityperson2coco(anno_data):
    global image_id, bbox_id
    json_dict = {"images": [], "annotations": [], "categories": []}
    anno_data = anno_data['anno_%s_aligned'%data_type].reshape(-1,)
    data_size = anno_data.shape[0]
    for ii in range(data_size):
        cell_data = anno_data[ii][0][0]
        file_name = cell_data[1][0]
        image = {'file_name': file_name, 'height': 1024, 'width': 2048, 'id': image_id}
        json_dict['images'].append(image)
        box_data = cell_data[2]
        box_data_size = box_data.shape[0]
        box_data = box_data.tolist()
        for jj in range(box_data_size):
            category = box_data[jj][0]
            if category not in categories:
                new_id = len(categories) + 1
                categories[category] = new_id
            category_id = categories[category]
            x, y, w, h = box_data[jj][1:1+4]
            fbox = [x, y, w, h]
            x_v, y_v, w_v, h_v = box_data[jj][6:6+4]
            vbox = [x_v, y_v, w_v, h_v]
            if box_type == 'full':
                bbox = fbox
            else:
                bbox = vbox
            annotation = {'area': fbox[2] * fbox[3], 'iscrowd': 0, 'image_id': image_id,
                         'bbox': bbox, 'category_id': category_id, 'id': bbox_id}
            json_dict['annotations'].append(annotation)
            bbox_id += 1
        image_id += 1
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    return json_dict

def save_result(res_dict):
    os.makedirs(new_anntation_dir, exist_ok=True)
    file_name = data_type + '.json'
    with open(os.path.join(new_anntation_dir, file_name), 'w') as f:
        json_str = json.dumps(res_dict)
        f.write(json_str)

if __name__ == '__main__':
    t1 = time.time()
    print("Converting [%s] cityperson dataset into COCO format..."%data_type)
    anno_data = load_annotation()
    res_dict = cityperson2coco(anno_data)
    save_result(res_dict)
    t2 = time.time()
    print("Finished in %g s" % (t2 - t1))
    

import os
from PIL import Image
import numpy as np
from collections import namedtuple

CityscapesClass= namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color', 'sim_category', 'sim_color'])
classes = [
    CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0), 'misc', (0, 0, 0)),
    CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0), 'misc', (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0), 'misc', (0, 0, 0)),
    CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0), 'misc', (0, 0, 0)),
    CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0), 'misc', (0, 0, 0)),
    CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0), 'misc', (0, 0, 0)),
    CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81), 'misc', (0, 0, 0)),
    CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128), 'ground', (55,55,55)),
    CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232), 'ground', (55,55,55)),
    CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160), 'misc', (0, 0, 0)),
    CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140), 'misc', (0, 0, 0)),
    CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70), 'building', (0,255,255)),
    CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156), 'building', (0,255,255)),
    CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153), 'building', (0,255,255)),
    CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180), 'misc', (0, 0, 0)),
    CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100), 'misc', (0, 0, 0)),
    CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90), 'misc', (0, 0, 0)),
    CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153), 'traffic items', (255,212,0)),
    CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153), 'misc', (0, 0, 0)),
    CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30), 'traffic items', (255,212,0)),
    CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0), 'traffic items', (255,212,0)),
    CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35), 'vegetation', (0,255,0)),
    CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152), 'terrain', (255, 97, 0)),
    CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180), 'sky', (8,19,49)),
    CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60), 'person*',  (220, 20, 60)),
    CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0), 'person*',  (220, 20, 60) ),
    CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142), 'car', (255, 0, 0)),
    CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70), 'truck*', (0, 80, 100)),
    CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100), 'truck*', (0, 80, 100)),
    CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90), 'truck*', (0, 80, 100)),
    CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110), 'truck*', (0, 80, 100)),
    CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100), 'truck*', (0, 80, 100)),
    CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230), 'motorcycle*', (119,11,32)),
    CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32), 'motorcycle*', (119,11,32)),
    CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142), 'misc', (0,0,0)),
]

train_id_to_sim_color = [c.sim_color for c in classes if (c.train_id != -1 and c.train_id != 255)]
train_id_to_sim_color.append((0, 0, 0))

def convert_cityscapes_sim(image_path, save_path):
    """
    path to image of size H*W, with encoding being the class id [0 - 19]
    """
    arr = np.array(Image.open(image_path))
    rgb_im = np.zeros((arr.shape[0], arr.shape[1], 3))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            rgb_im[i,j] = np.array(train_id_to_sim_color[arr[i,j]])
    Image.fromarray(rgb_im.astype(np.uint8)).save(os.path.join(save_path, os.path.basename(image_path)))

if __name__ == "__main__":
    save_path = '/network/tmp1/ccai/data/munit_dataset/non_flooded/streetview_mvp_seg_simlabels/'
    im_path = '/network/tmp1/ccai/data/munit_dataset/non_flooded/streetview_mvp_deeplabv3/'
    images_path =  [os.path.join(im_path,x) for x in os.listdir(im_path) if x.endswith('_pred.png')]
    for elem in images_path:
        convert_cityscapes_sim(elem, save_path)
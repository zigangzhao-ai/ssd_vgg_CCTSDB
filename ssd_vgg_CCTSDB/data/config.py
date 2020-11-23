# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("/workspace/zigangzhao/ssd_vgg/")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

USE_FL = False
USE_ECA = False
USE_SE = True
USE_CBAM = False
# SSD300 CONFIGS

voc = {
    'num_classes': 4,
    'lr_steps': (8000, 12000, 16000),
    'max_iter': 20000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300], 
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
 
    'aspect_ratios': [[2], [2, 3], [2, 3], [2,3],[2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

voc1 = {
    'num_classes': 4,
    'lr_steps': (8000, 12000, 16000),
    'max_iter': 20000,
    #'feature_maps': [38, 19, 10, 5, 3, 1],
    'feature_maps': [38, 19, 10, 5],
    'min_dim': 300,
    #'steps': [8, 16, 32, 64, 100, 300],
    'steps': [8, 16, 32, 64],
    # 'min_sizes': [30, 60, 111, 162, 213, 264],
    # 'max_sizes': [60, 111, 162, 213, 264, 315],
    'min_sizes': [30, 60, 111, 162],
    'max_sizes': [60, 111, 162, 213],
    'aspect_ratios': [[1.8, 2.8], [1.8, 2.8], [1.8, 2.8], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}


coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

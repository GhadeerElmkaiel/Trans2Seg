"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .mscoco import COCOSegmentation
from .cityscapes import CitySegmentation
from .ade import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .sbu_shadow import SBUSegmentation
from .transparent11 import TransparentSegmentation
from .transparent11_boundary import TransparentSegmentationBoundary
from .trans10k_boundary import TransSegmentationBoundary
from .sber_dataset import SberSegmentation
from .sber_dataset_all_classes import SberSegmentationAll
from .sber_dataset_all_no_fu_classes import SberSegmentationAllNoFU
from .sber_new_dataset_all_classes import  SberNewSegmentationAll
from .sber_merged_dataset_all_classes import  SberMergedSegmentationAll
# sber_merged_dataset_all_classes

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'coco': COCOSegmentation,
    'trans10k_boundary': TransSegmentationBoundary,
    'cityscape': CitySegmentation,
    'sbu': SBUSegmentation,
    'transparent11': TransparentSegmentation, 
    'transparent11_boundary': TransparentSegmentationBoundary,
    'sber_dataset': SberSegmentation,
    'sber_dataset_all': SberSegmentationAll,
    'sber_dataset_all_no_fu': SberSegmentationAllNoFU,
    'sber_merged_dataset_all': SberMergedSegmentationAll,
    'sber_new_dataset_all': SberNewSegmentationAll,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)

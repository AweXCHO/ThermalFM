from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp

@DATASETS.register_module()
class SCUTdataset(BaseSegDataset):
    """Thermal dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes = ('unlabelled', 'road', 'person', 'rider', 'car', 'truck', 'fence', 'tree', 'bus','pole'),

        palette = [[0, 0, 0], [128, 64, 128], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0], [153, 153, 190], [35, 142, 107], [100, 60, 0],[153, 153, 153]])
    

    def __init__(self, **kwargs):   
        super(SCUTdataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
            #img_suffix='.png', seg_map_suffix='.png', **kwargs)
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp

@DATASETS.register_module()
class MassMINDdataset(BaseSegDataset):
    """Thermal dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes = ('self', 'background', 'living obstacles', 'obstacles', 'bridge', 'water', 'sky'),

        palette = [[128, 64, 128], [0, 0, 0], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0], [153, 153, 190]])
    

    def __init__(self, **kwargs):   
        super(MassMINDdataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
            #img_suffix='.png', seg_map_suffix='.png', **kwargs)
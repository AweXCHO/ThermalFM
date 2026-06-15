from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp

@DATASETS.register_module()
class GASdataset(BaseSegDataset):
    """Thermal dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes = ('unlabelled', 'gas'),

        palette = [[0, 0, 0], [128, 64, 128]])
    

    def __init__(self, **kwargs):   
        super(GASdataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', reduce_zero_label=False,
            ignore_index=10,**kwargs)
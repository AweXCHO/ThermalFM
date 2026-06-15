from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp

@DATASETS.register_module()
class MSRSPersondataset(BaseSegDataset):
    """Thermal dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes = ('unlabelled', 'person'),

        palette = [[0, 0, 0], [64, 0, 128]])
    

    def __init__(self, **kwargs):
        super(MSRSPersondataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp

@DATASETS.register_module()
class MassMINDSingledataset(BaseSegDataset):
    """Thermal dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes = ('unlabelled', 'obstacles'),

        palette = [[0, 0, 0], [128, 64, 128]])
    

    def __init__(self, **kwargs):   
        super(MassMINDSingledataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', **kwargs)
            #img_suffix='.png', seg_map_suffix='.png', **kwargs)
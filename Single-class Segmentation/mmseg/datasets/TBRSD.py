from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp


@DATASETS.register_module()
class TBRSDDataset(BaseSegDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes = ('background', 'blind'),

        palette = [[0,0,0],[60, 20, 220]])

    def __init__(self, **kwargs):
        super(TBRSDDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
# The code for each network is migrated from their code base
# DORN: https://github.com/liviniuk/DORN_depth_estimation_Pytorch
# BTS: https://github.com/cleinc/bts
# MiDaS: https://github.com/isl-org/MiDaS
# AdaBins: https://github.com/shariqfarooq123/AdaBins
# NeWCRF: https://github.com/aliyun/NeWCRFs

# PSMNet: https://github.com/JiaRenChang/PSMNet
# GWCNet: https://github.com/xy-guo/GwcNet
# CFNet: https://github.com/gallenszl/CFNet
# AANet: https://github.com/haofeixu/aanet
# ACVNet: https://github.com/gangweiX/ACVNet

# monocular depth network
from .dorn.dorn import DeepOrdinalRegression
from .bts.bts import BtsModel
from .midas import DPTDepthModel, MidasNet, MidasNet_small
from .adabin import UnetAdaptiveBins
from .newcrf import NewCRFDepthMyswin, NewCRFDepth, NewCRFDepthInfMAE, NewCRFDepthDinov3, NewCRFDepth_myswin_old
from .kitti import _make_dinov2_linear_depther
from .kitti import _make_dinov2_dpt_depther
from .vifl import myMonodepth, DHRNet, LiteMono
from .single import mySimple
from .depthpro import DepthProNetwork
from .depthfm import DepthFMNet
from .dpt import SwinMAEWithDPT
'''
# stereo depth network
from .aanet import AANetModel
from .psmnet import PSMNetModel
from .acvnet import ACVNetModel
from .cfnet import CFNetModel
from .gwcnet import GWCNetModel_G, GWCNetModel_GC
from .ms_crf import MonoStereoCRFDepth
'''
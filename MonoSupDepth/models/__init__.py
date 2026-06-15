from .registry import MODELS

# Supervised Monocular Depth Network 
from .trainers.mono_depth.DORN import DORN
from .trainers.mono_depth.BTS import BTS
from .trainers.mono_depth.AdaBins import AdaBins
from .trainers.mono_depth.Midas import Midas
from .trainers.mono_depth.NewCRF import NewCRF
from .trainers.mono_depth.NewCRF_Myswin import NewCRFMyswin
from .trainers.mono_depth.KITTI import KITTI
from .trainers.mono_depth.VIFL import VIFL
from .trainers.mono_depth.Single import Single
from .trainers.mono_depth.DepthPro import DepthPro
from .trainers.mono_depth.DepthFM import DepthFM
from .trainers.mono_depth.SwinwithDPT import DPT
from .trainers.mono_depth.NewCRF_Dino import NewCRFDino
from .trainers.mono_depth.NewCRF_Myswin_old import NewCRFMyswin_old
# Supervised Stereo Matching Network 
'''
from .trainers.stereo_depth.PSMNet import PSMNet
from .trainers.stereo_depth.AANet import AANet
from .trainers.stereo_depth.GWCNet import GWCNet
from .trainers.stereo_depth.CFNet import CFNet
from .trainers.stereo_depth.ACVNet import ACVNet
from .trainers.stereo_depth.MonoStereoCRF import MonoStereoCRF'''
 
# custom_encoders.py
import segmentation_models_pytorch as smp
from timm_swin_mae import SwinMAE

encoder_name = "SwinMAE"

smp.encoders.encoders[encoder_name] = {
    "encoder": SwinMAE,
    "params": {
        "in_channels": 3,
    }
}

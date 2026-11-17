"""Fast CPU checks for the extracted package and checkpoint compatibility."""

import unittest

import torch

from AAA_backbone import ThermalFMPhysicsBackbone
from AAA_phys.swin_mae_feature_serial import SwinMAEFeatureSerial


def build_backbone(output_mode="post_stage"):
    return ThermalFMPhysicsBackbone(
        patch_size=4,
        depths=(1, 1, 1, 1),
        embed_dim=16,
        num_heads=(1, 2, 4, 8),
        window_size=2,
        drop_path_rate=0.0,
        output_mode=output_mode,
    )


def build_pretrain():
    return SwinMAEFeatureSerial(
        img_size=64,
        patch_size=4,
        mask_ratio=0.0,
        in_chans=3,
        decoder_embed_dim=128,
        depths=(1, 1, 1, 1),
        embed_dim=16,
        num_heads=(1, 2, 4, 8),
        window_size=2,
        drop_path_rate=0.0,
        mask_window_size=2,
    )


class ExtractedBackboneTest(unittest.TestCase):
    def test_feature_pyramid_shapes(self):
        model = build_backbone("pre_downsample").eval()
        outputs = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(
            [tuple(item.shape) for item in outputs],
            [(2, 16, 16, 16), (2, 32, 8, 8), (2, 64, 4, 4), (2, 128, 2, 2)],
        )

    def test_non_square_input(self):
        model = build_backbone("pre_downsample").eval()
        outputs = model(torch.randn(1, 3, 61, 77))
        self.assertEqual(tuple(outputs[0].shape), (1, 16, 16, 20))
        self.assertEqual(tuple(outputs[-1].shape), (1, 128, 2, 3))

    def test_exact_unmasked_encoder_equivalence(self):
        torch.manual_seed(7)
        pretrain = build_pretrain().eval()
        model = build_backbone().eval()
        report = model.load_pretrained({"model": pretrain.state_dict()})
        self.assertGreater(report.loaded_count, 0)
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            expected, mask = pretrain.forward_encoder(images)
            actual = model(images)[-1].permute(0, 2, 3, 1)
        self.assertEqual(float(mask.max()), 0.0)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()

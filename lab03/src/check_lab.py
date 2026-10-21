"""Focused checks for architecture, patch layout, gradients, and profiler coverage."""
from __future__ import annotations

import unittest
from copy import deepcopy

import torch
from thop import profile
from torch import nn

from costs import count_mha_macs, hand_costs, profile_model, visible_attention_operations, zero_ops
from models import SPECS, PatchAttention, build_model


class ArchitectureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_exact_parameter_and_flop_counts(self):
        expected = {
            "reference_mlp": (235_146, 469_504),
            "reference_cnn": (20_490, 2_063_488),
            "reference_attention": (70_794, 2_329_856),
            "matched_mlp": (118_282, 236_032),
            "matched_cnn": (118_634, 22_478_848),
            "matched_attention": (119_794, 3_918_096),
        }
        for spec in SPECS:
            with self.subTest(model=spec.key):
                model = build_model(spec)
                hand = hand_costs(spec)
                parameters = sum(parameter.numel() for parameter in model.parameters())
                self.assertEqual((parameters, sum(row["hand_flops"] for row in hand)), expected[spec.key])
                self.assertEqual(sum(row["hand_parameters"] for row in hand), parameters)
                if spec.suite == "matched":
                    self.assertTrue(100_000 <= parameters <= 150_000)

    def test_forward_backward_and_batch_independence(self):
        for spec in SPECS:
            with self.subTest(model=spec.key):
                model = build_model(spec)
                images = torch.randn(2, 1, 28, 28)
                logits = model(images)
                self.assertEqual(tuple(logits.shape), (2, 10))
                nn.functional.cross_entropy(logits, torch.tensor([1, 4])).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                model.eval()
                with torch.inference_mode():
                    batched = model(images)
                    separate = torch.cat([model(image.unsqueeze(0)) for image in images])
                torch.testing.assert_close(batched, separate, atol=1e-5, rtol=1e-5)

    def test_patch_layout_and_coverage(self):
        images = torch.arange(2 * 784).reshape(2, 1, 28, 28)
        patches = PatchAttention.patchify(images)
        self.assertEqual(tuple(patches.shape), (2, 16, 49))
        for batch in range(2):
            for row in range(4):
                for column in range(4):
                    expected = images[batch, :, row * 7:(row + 1) * 7,
                                      column * 7:(column + 1) * 7].reshape(49)
                    torch.testing.assert_close(patches[batch, row * 4 + column], expected)
            torch.testing.assert_close(patches[batch].reshape(-1).sort().values,
                                       images[batch].reshape(-1))

    def test_attention_patch_permutation_invariance_and_independent_layers(self):
        model = PatchAttention().eval()
        self.assertFalse(torch.equal(model.encoders[0].linear1.weight, model.encoders[1].linear1.weight))
        images = torch.randn(2, 1, 28, 28)
        patches = model.patchify(images)[:, torch.randperm(16)]
        permuted_images = patches.reshape(2, 4, 4, 1, 7, 7).permute(0, 3, 1, 4, 2, 5).reshape(2, 1, 28, 28)
        with torch.inference_mode():
            torch.testing.assert_close(model(images), model(permuted_images), atol=2e-6, rtol=1e-5)

    def test_profiler_covers_attention_and_restores_original_model(self):
        for spec in SPECS:
            with self.subTest(model=spec.key):
                model = build_model(spec)
                original = deepcopy(model.state_dict())
                fastpath = torch.backends.mha.get_fastpath_enabled()
                counts, _, _ = profile_model(model, spec, torch.randn(1, 1, 28, 28))
                self.assertEqual(counts["hand_flops"], counts["thop_core_flops"])
                self.assertEqual(counts["parameters"], counts["thop_core_parameters"])
                self.assertEqual(torch.backends.mha.get_fastpath_enabled(), fastpath)
                self.assertEqual(set(model.state_dict()), set(original))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, original[name], atol=0, rtol=0)
                custom = {nn.MultiheadAttention: count_mha_macs, nn.LayerNorm: zero_ops}
                with visible_attention_operations():
                    batch_macs, _ = profile(deepcopy(model), inputs=(torch.randn(4, 1, 28, 28),),
                                            custom_ops=custom, verbose=False)
                self.assertEqual(int(2 * batch_macs), 4 * counts["hand_flops"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Mechanism checks for the tiny ECG JEPA loop. No PTB-XL download required."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ecg_jepa.config import Config
from ecg_jepa.downstream import positive_subset_auc, remove_head_directions
from ecg_jepa.data.ptbxl import (
    SUPERCLASSES,
    SyntheticECG,
    encode_named_labels,
    encode_superclasses,
    fit_length,
    make_dataloader,
    resolve_ptbxl_root,
    select_independent_leads,
    zscore_leads,
)
from ecg_jepa.probe import macro_auc, roc_auc
from ecg_jepa.robustness import sample_mean_std
from ecg_jepa.amplitude import (
    TwoBranch,
    amplitude_features,
    amplitude_helps,
    head_checkpoint_path,
    predict_scores,
    prepare_recording,
    save_two_branch,
    standardize,
    threshold_report,
)
from ecg_jepa.finetune import EncoderClassifier, finetune_split, unfreeze
from ecg_jepa.hyp_voltage import (
    beats_embedding,
    hyp_bucket,
    hyp_gap_reason,
    hyp_subclasses,
    peak_r_amplitude,
)
from ecg_jepa.low_label import budget_count, budget_indices, clears_seed_spread
from ecg_jepa.models.jepa import (
    JEPA,
    block_length_bounds,
    ema_momentum,
    ema_update,
    sample_multiblock_mask,
)
from ecg_jepa.train import run, set_seed


def _small_cfg(**overrides) -> Config:
    cfg = Config(
        enc_dim=64,
        enc_depth=2,
        enc_heads=4,
        pred_dim=32,
        pred_depth=2,
        pred_heads=4,
        batch_size=8,
        epochs=1,
        log_every=1,
        weight_decay=0.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class DataTests(unittest.TestCase):
    def test_independent_leads_drop_derived_channels(self):
        names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        signal = np.arange(12, dtype=np.float32).reshape(1, 12)
        chosen = select_independent_leads(signal, names)
        self.assertEqual(chosen.shape, (8, 1))
        self.assertTrue(np.allclose(chosen[:, 0], [0, 1, 6, 7, 8, 9, 10, 11]))

    def test_zscore_is_per_lead_and_finite_when_flat(self):
        signal = np.ones((8, 20), dtype=np.float32)
        signal[1] = np.linspace(-1, 1, 20, dtype=np.float32)
        scored = zscore_leads(signal)
        self.assertTrue(np.allclose(scored[0], 0.0))
        self.assertAlmostEqual(float(scored[1].mean()), 0.0, places=5)
        self.assertTrue(np.isfinite(scored).all())

    def test_fit_length_crops_and_pads(self):
        signal = np.arange(16, dtype=np.float32).reshape(2, 8)
        cropped = fit_length(signal, 4)
        self.assertEqual(cropped.shape, (2, 4))
        self.assertTrue(np.allclose(cropped[0], [2, 3, 4, 5]))
        padded = fit_length(signal, 10)
        self.assertEqual(padded.shape, (2, 10))
        self.assertTrue(np.allclose(padded[0, 1:9], signal[0]))

    def test_resolve_accepts_nested_kaggle_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
            nested.mkdir()
            (nested / "ptbxl_database.csv").write_text("ecg_id\n", encoding="utf-8")
            self.assertEqual(resolve_ptbxl_root(root), nested)
            self.assertEqual(resolve_ptbxl_root(nested), nested)

    def test_named_labels_and_unused_embedding_directions(self):
        labels = encode_named_labels("{'AMI': 100.0, 'SR': 0.0}", ["AMI", "IMI"])
        self.assertEqual(labels.tolist(), [1.0, 0.0])
        subclass = encode_named_labels(
            "{'AMI': 100.0, 'LVH': 50.0}",
            ["AMI", "LVH", "NORM"],
            {"AMI": "AMI", "LVH": "LVH"},
        )
        self.assertEqual(subclass.tolist(), [1.0, 1.0, 0.0])
        features = torch.randn(30, 8)
        weight = torch.randn(3, 8)
        residual = remove_head_directions(features, weight)
        self.assertTrue(torch.allclose(residual @ weight.T, torch.zeros(30, 3), atol=1e-4))
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.05])
        target = np.array([0, 0, 1, 1, 1])
        pure = np.array([False, False, True, False, False])
        mixed = np.array([False, False, False, True, True])
        pure_auc, n_pure = positive_subset_auc(target, scores, pure)
        _mixed_auc, n_mixed = positive_subset_auc(target, scores, mixed)
        self.assertEqual((n_pure, n_mixed), (1, 2))
        self.assertGreater(pure_auc, 0.9)

    def test_superclass_encoding_keeps_diagnostic_keys(self):
        mapping = {"NORM": "NORM", "AMI": "MI", "SR": "RHYTHM", "LVH": "HYP"}
        labels = encode_superclasses("{'NORM': 100.0, 'AMI': 0.0, 'SR': 80.0, 'LVH': 50.0}", mapping)
        self.assertEqual(labels.tolist(), [1.0, 1.0, 0.0, 0.0, 1.0])

    def test_roc_auc_perfect_and_tied(self):
        labels = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
        self.assertAlmostEqual(roc_auc(labels, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)
        score, per_class = macro_auc(
            np.array([[0, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 0]]),
            np.array(
                [
                    [0.1, 0.9, 0.2, 0.1, 0.8],
                    [0.8, 0.2, 0.9, 0.2, 0.1],
                    [0.7, 0.6, 0.7, 0.9, 0.7],
                    [0.2, 0.1, 0.1, 0.8, 0.2],
                ]
            ),
        )
        self.assertGreater(score, 0.9)
        self.assertEqual(len(per_class), 5)

    def test_sample_mean_std_uses_sample_variance(self):
        mean, std = sample_mean_std([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, 1.0)

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            make_dataloader(Config(), "holdout", synthetic=True, data_dir=None)


class JEPATests(unittest.TestCase):
    def test_context_encoder_excludes_masked_indices(self):
        cfg = _small_cfg()
        model = JEPA(cfg)
        batch = torch.stack([SyntheticECG(4, cfg.n_leads, cfg.signal_length, seed=1)[i] for i in range(4)])
        captured = {}

        context_forward = model.context_encoder.forward
        target_forward = model.target_encoder.forward

        def spy_context(x, keep_idx=None):
            captured["keep_idx"] = None if keep_idx is None else keep_idx.detach().cpu()
            out = context_forward(x, keep_idx)
            captured["context_len"] = out.shape[1]
            return out

        def spy_target(x, keep_idx=None):
            captured["target_keep"] = keep_idx
            out = target_forward(x, keep_idx)
            captured["target_len"] = out.shape[1]
            return out

        model.context_encoder.forward = spy_context
        model.target_encoder.forward = spy_target
        out = model(batch)

        keep = captured["keep_idx"].tolist()
        mask = out.mask_idx.detach().cpu().tolist()
        self.assertIsNone(captured["target_keep"])
        self.assertEqual(captured["target_len"], cfg.n_patches)
        self.assertEqual(captured["context_len"], len(keep))
        self.assertTrue(set(keep).isdisjoint(mask))
        self.assertEqual(set(keep) | set(mask), set(range(cfg.n_patches)))

    def test_finetune_updates_encoder_and_keeps_best_fold9_epoch(self):
        cfg = _small_cfg(signal_length=40, patch_size=20, enc_dim=32, pred_dim=32)
        encoder = JEPA(cfg).target_encoder
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        before = encoder.pos_embed.detach().clone()
        train_x = torch.randn(16, cfg.n_leads, cfg.signal_length)
        val_x = torch.randn(8, cfg.n_leads, cfg.signal_length)
        test_x = torch.randn(8, cfg.n_leads, cfg.signal_length)
        train_y = torch.randint(0, 2, (16, 5)).float()
        val_y = torch.randint(0, 2, (8, 5)).float()
        test_y = torch.randint(0, 2, (8, 5)).float()
        result = finetune_split(
            encoder, train_x, train_y, val_x, val_y, test_x, test_y, seed=0, device=torch.device("cpu"), epochs=2
        )
        self.assertFalse(torch.equal(encoder.pos_embed.detach(), before))
        self.assertEqual(result["best_epoch"], max(range(2), key=result["val_curve"].__getitem__))
        self.assertEqual(unfreeze(encoder).pos_embed.requires_grad, True)
        logits = EncoderClassifier(encoder, 5)(train_x[:2])
        self.assertEqual(tuple(logits.shape), (2, 5))

    def test_multiblock_mask_is_a_union_of_four_spans(self):
        low, high = block_length_bounds(40, 0.175, 0.225)
        self.assertEqual((low, high), (7, 9))
        torch.manual_seed(0)
        keep, mask = sample_multiblock_mask(40, 4, 0.175, 0.225, torch.device("cpu"))
        self.assertEqual(set(keep.tolist()) | set(mask.tolist()), set(range(40)))
        self.assertTrue(set(keep.tolist()).isdisjoint(mask.tolist()))
        self.assertGreaterEqual(int(mask.numel()), low)
        self.assertLessEqual(int(mask.numel()), min(40 - 1, 4 * high))
        self.assertTrue(torch.equal(mask, torch.sort(mask).values))

    def test_multiblock_model_predicts_only_masked_times(self):
        cfg = _small_cfg(mask_mode="multiblock")
        model = JEPA(cfg)
        batch = torch.randn(2, cfg.n_leads, cfg.signal_length)
        out = model(batch)
        self.assertTrue(set(out.keep_idx.tolist()).isdisjoint(out.mask_idx.tolist()))
        self.assertEqual(out.pred.shape[1], out.mask_idx.numel())
        self.assertGreaterEqual(out.mask_idx.numel(), 7)

    def test_per_lead_mask_drops_every_lead_at_hidden_times(self):
        cfg = _small_cfg(token_mode="per_lead", lead_dim=32)
        model = JEPA(cfg)
        batch = torch.randn(2, cfg.n_leads, cfg.signal_length)
        captured = {}
        lead_forward = model.context_encoder.lead_mixer.forward

        def spy(tokens):
            captured["leads"] = tokens.shape[1]
            captured["times"] = tokens.shape[2]
            return lead_forward(tokens)

        model.context_encoder.lead_mixer.forward = spy
        out = model(batch)
        self.assertEqual(captured["leads"], cfg.n_leads)
        self.assertEqual(captured["times"], out.keep_idx.numel())
        self.assertEqual(out.pred.shape[-1], cfg.enc_dim)
        self.assertTrue(set(out.keep_idx.tolist()).isdisjoint(out.mask_idx.tolist()))

    def test_prediction_matches_masked_target_shape(self):
        cfg = _small_cfg()
        model = JEPA(cfg)
        batch = torch.randn(4, cfg.n_leads, cfg.signal_length)
        out = model(batch)
        self.assertEqual(out.pred.shape, out.target.shape)
        self.assertEqual(out.pred.shape[0], 4)
        self.assertEqual(out.pred.shape[1], out.mask_idx.numel())
        self.assertEqual(out.pred.shape[-1], cfg.enc_dim)
        self.assertEqual(out.keep_idx.numel() + out.mask_idx.numel(), cfg.n_patches)
        self.assertEqual(out.loss.ndim, 0)

    def test_target_encoder_receives_no_gradient(self):
        cfg = _small_cfg()
        model = JEPA(cfg)
        batch = torch.randn(2, cfg.n_leads, cfg.signal_length)
        out = model(batch)
        out.loss.backward()
        for param in model.target_encoder.parameters():
            self.assertFalse(param.requires_grad)
            self.assertIsNone(param.grad)
        self.assertTrue(
            any(p.grad is not None and torch.any(p.grad != 0) for p in model.context_encoder.parameters())
        )
        self.assertTrue(
            any(p.grad is not None and torch.any(p.grad != 0) for p in model.predictor.parameters())
        )

    def test_overfit_one_batch_loss_falls_without_collapse(self):
        cfg = _small_cfg()
        set_seed(0)
        dataset = SyntheticECG(8, cfg.n_leads, cfg.signal_length, seed=0)
        batch = torch.stack([dataset[i] for i in range(8)])
        model = JEPA(cfg)
        optimizer = torch.optim.AdamW(
            list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
            lr=cfg.lr,
            weight_decay=0.0,
        )
        steps = 40
        losses = []
        stds = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            out.loss.backward()
            optimizer.step()
            ema_update(
                model.target_encoder,
                model.context_encoder,
                ema_momentum(step, steps, cfg.ema_start, cfg.ema_end),
            )
            losses.append(out.loss.item())
            stds.append(out.embed_std.item())
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        self.assertLess(late, early)
        self.assertGreater(min(stds), 1e-3)

    def test_one_step_synthetic_train_writes_checkpoint(self):
        cfg = _small_cfg(batch_size=4, epochs=1, log_every=1)
        with tempfile.TemporaryDirectory() as tmp:
            cfg.ckpt_dir = tmp
            path = run(cfg, synthetic=True, max_steps=1, device="cpu", val_batches=1)
            self.assertTrue(Path(path).is_file())
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("context_encoder", saved)
            self.assertIn("target_encoder", saved)
            self.assertEqual(saved["step"], 1)


class AmplitudeTests(unittest.TestCase):
    def test_amplitude_features_are_per_lead_and_ignore_zscore_scale(self):
        signal = np.zeros((8, 4), dtype=np.float32)
        signal[0] = np.array([0, 0, 0, 4], dtype=np.float32)
        signal[6, 1] = 2
        features = amplitude_features(signal)
        self.assertEqual(features.shape, (24,))
        self.assertAlmostEqual(float(features[0]), 4.0)
        self.assertAlmostEqual(float(features[6]), 2.0)
        self.assertAlmostEqual(float(features[8]), float(np.std(signal[0])))
        self.assertAlmostEqual(float(features[16]), 4.0)
        self.assertAlmostEqual(float(features[22]), 2.0)
        self.assertAlmostEqual(peak_r_amplitude(signal), 2.0)

    def test_scaler_uses_the_training_fold_only(self):
        train = torch.tensor([[0.0], [2.0], [4.0]])
        held_out = torch.tensor([[100.0]])
        scaled_train, scaled_held = standardize(train, held_out)
        self.assertAlmostEqual(float(scaled_train.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(scaled_train[0]), float((0.0 - 2.0) / train.std(unbiased=False)))
        self.assertGreater(float(scaled_held), 5.0)

    def test_two_branch_trains_the_amplitude_path_only(self):
        model = TwoBranch(embed_dim=8, n_amplitude=4, n_classes=5, width=8)
        loss = model(torch.randn(3, 8), torch.randn(3, 4)).sum()
        loss.backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertIsNotNone(model.amplitude[0].weight.grad)
        self.assertEqual(tuple(model(torch.randn(2, 8), torch.randn(2, 4)).shape), (2, 5))

    def test_branch_runs_only_when_the_macro_gap_clears_the_spread(self):
        published = [0.006, 0.001, -0.002, 0.007, 0.003]
        self.assertFalse(amplitude_helps(published))
        self.assertTrue(amplitude_helps([0.02, 0.018, 0.021, 0.019, 0.022]))

    def test_threshold_half_counts_as_positive(self):
        labels = np.array(
            [
                [1, 0, 1, 0, 0],
                [1, 1, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        scores = np.array(
            [
                [0.9, 0.2, 0.8, 0.1, 0.4],
                [0.5, 0.7, 0.4, 0.2, 0.8],
                [0.4, 0.9, 0.1, 0.6, 0.2],
                [0.1, 0.3, 0.7, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        report = threshold_report(labels, scores, threshold=0.5)
        self.assertEqual(set(report), set(SUPERCLASSES))
        for name in SUPERCLASSES:
            self.assertAlmostEqual(report[name]["accuracy"], 1.0)
            self.assertAlmostEqual(report[name]["f1"], 1.0)
        scores[0, 0] = 0.49
        missed = threshold_report(labels, scores, threshold=0.5)
        self.assertAlmostEqual(missed["NORM"]["accuracy"], 0.75)
        self.assertAlmostEqual(missed["NORM"]["f1"], 2 / 3)

    def test_saved_head_scores_one_ecg_and_leaves_the_encoder_file(self):
        cfg = _small_cfg(signal_length=40, patch_size=20, enc_dim=32)
        encoder = JEPA(cfg).target_encoder
        signal = np.random.default_rng(0).normal(size=(8, 50)).astype(np.float32)
        signal += np.linspace(-0.2, 0.4, 8, dtype=np.float32)[:, None]
        mean = torch.zeros(24)
        std = torch.ones(24)
        wave, scaled = prepare_recording(signal, cfg.signal_length, mean, std)
        doubled_wave, doubled_scaled = prepare_recording(signal * 2, cfg.signal_length, mean, std)
        self.assertTrue(torch.allclose(wave, doubled_wave, atol=1e-5))
        self.assertFalse(torch.allclose(scaled, doubled_scaled))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder_path = root / "last.pt"
            torch.save(
                {"config": cfg.__dict__, "target_encoder": encoder.state_dict()},
                encoder_path,
            )
            original = encoder_path.read_bytes()
            head = TwoBranch(cfg.enc_dim, 24, 5, width=8)
            path = save_two_branch(encoder_path, head, mean, std, seed=0, signal_length=cfg.signal_length)
            self.assertEqual(path, head_checkpoint_path(encoder_path))
            self.assertEqual(path.name, "two_branch.pt")
            scores = predict_scores(signal, path, device="cpu")
            self.assertEqual(list(scores), list(SUPERCLASSES))
            for value in scores.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
            self.assertEqual(encoder_path.read_bytes(), original)
            with self.assertRaises(ValueError):
                predict_scores(signal.T, path, device="cpu")


class BudgetTests(unittest.TestCase):
    def test_percent_budgets_are_stable_shares_of_folds_1_to_8(self):
        self.assertEqual(budget_count(17441, 1), 174)
        self.assertEqual(budget_count(17441, 10), 1744)
        one = budget_indices(17441, 1, seed=0)
        ten = budget_indices(17441, 10, seed=0)
        self.assertEqual(len(one), 174)
        self.assertEqual(len(ten), 1744)
        self.assertTrue(np.array_equal(one, budget_indices(17441, 1, seed=0)))
        self.assertFalse(np.array_equal(one, budget_indices(17441, 1, seed=1)))
        self.assertFalse(np.array_equal(one, ten[:174]))
        self.assertGreaterEqual(int(one[0]), 0)
        self.assertLess(int(one[-1]), 17441)
        self.assertEqual(len(np.unique(one)), 174)
        self.assertTrue(np.all(one[1:] > one[:-1]))

    def test_full_label_finetune_gap_does_not_clear_the_seed_spread(self):
        published = [0.006, 0.001, -0.002, 0.007, 0.003]
        self.assertFalse(clears_seed_spread(published))
        self.assertTrue(clears_seed_spread([0.05, 0.04, 0.06, 0.05, 0.04]))
        self.assertFalse(clears_seed_spread([0.02]))


class HypVoltageTests(unittest.TestCase):
    def test_peak_r_is_the_tallest_precordial_sample(self):
        signal = np.zeros((8, 40), dtype=np.float32)
        signal[0, 10] = 5.0
        signal[6, 20] = 2.5
        signal[2, 4] = -3.0
        self.assertAlmostEqual(peak_r_amplitude(signal), 2.5)

    def test_hyp_buckets_name_lvh_rvh_and_the_rest(self):
        code_to_class = {"LVH": "HYP", "RVH": "HYP", "LAO/LAE": "HYP", "NORM": "NORM"}
        code_to_subclass = {"LVH": "LVH", "RVH": "RVH", "LAO/LAE": "LAO/LAE", "NORM": "NORM"}
        hits = hyp_subclasses("{'LVH': 100.0, 'NORM': 0.0}", code_to_class, code_to_subclass)
        self.assertEqual(hits, {"LVH"})
        self.assertEqual(hyp_bucket({"LVH", "LAO/LAE"}), "LVH")
        self.assertEqual(hyp_bucket({"RVH"}), "RVH")
        self.assertEqual(hyp_bucket({"LVH", "RVH"}), "LVH+RVH")
        self.assertEqual(hyp_bucket({"LAO/LAE", "SEHYP"}), "rest")

    def test_peak_r_beats_the_embedding_only_above_every_seed(self):
        seeds = [0.556, 0.582]
        self.assertTrue(beats_embedding(0.60, seeds))
        self.assertFalse(beats_embedding(0.57, seeds))
        self.assertFalse(beats_embedding(0.582, seeds))
        self.assertEqual(hyp_gap_reason(True, True), "temporal")
        self.assertEqual(hyp_gap_reason(True, False), "scale")
        self.assertEqual(hyp_gap_reason(False, True), "temporal_zscore")
        self.assertEqual(hyp_gap_reason(False, False), "sample_size")


if __name__ == "__main__":
    unittest.main()

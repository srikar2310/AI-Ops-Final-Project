"""Unit tests for model.py: make_inputs (the model's input contract) and RatingCNN.

Case IDs (UT-M-xx) match docs/test_cases.md.
"""
import numpy as np
import pytest

import phase7_helpers as h

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from features import N_FEATURES, build_sample, game_features  # noqa: E402
from model import IN_CHANNELS, RatingCNN, build_model, make_inputs  # noqa: E402

L = 32
SMALL = dict(channels=(8, 8), kernel=3, dilations=(1, 2), dropout=0.0, hidden=8, feat_hidden=8)


def batch(n_games=3, seed=0):
    wp, side, length, feats = [], [], [], []
    for g in range(n_games):
        cp = h.oscillating_evals(10 + 7 * g, seed=seed + g)
        for s in (0, 1):
            row, n = build_sample(cp, s, L, 1500)
            wp.append(row)
            side.append(s)
            length.append(n)
            feats.append(game_features(cp, s, "1-0", 1500))
    return (torch.tensor(np.stack(wp)), torch.tensor(side), torch.tensor(length),
            torch.tensor(np.stack(feats)))


# --------------------------------------------------------------- make_inputs
def test_um01_shapes():
    wp, side, length, _ = batch()
    x, mask = make_inputs(wp, side, length)
    assert x.shape == (wp.shape[0], IN_CHANNELS, L) and mask.shape == wp.shape


def test_um02_padding_is_zero_in_every_channel():
    wp, side, length, _ = batch()
    x, mask = make_inputs(wp, side, length)
    for i, n in enumerate(length.tolist()):
        assert torch.all(x[i, :, n:] == 0)
        assert torch.all(mask[i, :n] == 1)


def test_um03_own_move_channel_follows_side():
    wp = torch.full((2, 6), 0.5)
    x, _ = make_inputs(wp, torch.tensor([0, 1]), torch.tensor([6, 6]))
    assert x[0, 4].tolist() == [1, 0, 1, 0, 1, 0]      # White plays even plies
    assert x[1, 4].tolist() == [0, 1, 0, 1, 0, 1]


def test_um04_losses_are_attributed_and_non_negative():
    # player's view: White drops 0.5 -> 0.3 on ply 0 (own move), then rises on ply 1
    wp = torch.tensor([[0.3, 0.6, 0.6, 0.6]])
    x, _ = make_inputs(wp, torch.tensor([0]), torch.tensor([4]))
    own_loss, opp_loss = x[0, 2], x[0, 3]
    assert own_loss[0].item() == pytest.approx(0.2 * 5.0)
    assert opp_loss[1].item() == pytest.approx(0.3 * 5.0)    # gain on opponent's move
    assert torch.all(own_loss >= 0) and torch.all(opp_loss >= 0)
    assert torch.all(own_loss[1::2] == 0) and torch.all(opp_loss[0::2] == 0)


def test_um05_centred_channel():
    x, _ = make_inputs(torch.tensor([[0.75, 0.25]]), torch.tensor([0]), torch.tensor([2]))
    assert x[0, 0].tolist() == pytest.approx([0.5, -0.5])


# ----------------------------------------------------------------- RatingCNN
def model(**kw):
    torch.manual_seed(0)
    m = RatingCNN(**{**SMALL, **kw})
    m.eval()
    return m


def test_um06_forward_shape_and_finite():
    wp, side, length, feats = batch()
    out = model()(wp, side, length, feats)
    assert out.shape == (wp.shape[0],) and torch.all(torch.isfinite(out))


def test_um07_padding_content_does_not_change_output():
    wp, side, length, feats = batch()
    m = model()
    noisy = wp.clone()
    for i, n in enumerate(length.tolist()):
        noisy[i, n:] = torch.rand(L - n)
    torch.testing.assert_close(m(wp, side, length, feats), m(noisy, side, length, feats))


def test_um08_eval_mode_is_batch_independent():
    wp, side, length, feats = batch()
    m = model()
    full = m(wp, side, length, feats)
    single = torch.cat([m(wp[i:i + 1], side[i:i + 1], length[i:i + 1], feats[i:i + 1])
                        for i in range(len(wp))])
    torch.testing.assert_close(full, single, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kw,msg", [
    (dict(channels=(8, 8), dilations=(1,)), "same length"),
    (dict(kernel=4), "odd"),
])
def test_um09_invalid_architecture_is_rejected(kw, msg):
    with pytest.raises(ValueError, match=msg):
        RatingCNN(**{**SMALL, **kw})


def test_um10_features_required_when_enabled():
    wp, side, length, _ = batch()
    with pytest.raises(ValueError, match="feats"):
        model()(wp, side, length, None)


def test_um11_cnn_only_model_ignores_features():
    wp, side, length, _ = batch()
    out = model(use_features=False)(wp, side, length)
    assert out.shape == (wp.shape[0],)


def test_um12_feature_stats_guard_constant_columns_and_are_saved():
    m = model()
    std = np.ones(N_FEATURES)
    std[5] = 0.0
    mask = np.ones(N_FEATURES)
    mask[3] = 0
    m.set_feature_stats(np.arange(N_FEATURES), std, mask)
    assert m.feat_std[5].item() == 1.0              # std 0 replaced by 1
    state = m.state_dict()
    for key in ("feat_mean", "feat_std", "feat_mask"):
        assert key in state                          # travel with the weights
    m2 = model()
    m2.load_state_dict(state)
    assert m2.feat_mask[3].item() == 0 and m2.feat_mean[7].item() == 7


def test_um13_feature_mask_switches_a_feature_off():
    wp, side, length, feats = batch()
    m = model()
    mask = np.ones(N_FEATURES)
    mask[0] = 0
    m.set_feature_stats(np.zeros(N_FEATURES), np.ones(N_FEATURES), mask)
    changed = feats.clone()
    changed[:, 0] += 100.0
    torch.testing.assert_close(m(wp, side, length, feats), m(wp, side, length, changed))


def test_um14_build_model_from_params_yaml():
    with open(h.REPO / "params.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["train"]
    m = build_model(cfg)
    assert len(m.blocks) == len(cfg["channels"])
    assert m.use_features == bool(cfg.get("use_features", True))

"""Hybrid 1D CNN + feature network that predicts a player's rating.

Two inputs per sample:
  * the eval SEQUENCE of the game (both players, interleaved), read by the CNN
  * a vector of game-level FEATURES (move quality of both players, phase
    accuracy, conversion of winning positions, the result), read by a small MLP

Kept separate from train.py so the same code can be imported by the serving
API. `make_inputs` is part of the model contract: anything that feeds the
model (training, FastAPI) must go through it.
"""
import torch
import torch.nn as nn

from features import N_FEATURES

DELTA_SCALE = 5.0  # win-probability changes are small; scale them up a bit
IN_CHANNELS = 6


def make_inputs(wp, side, length):
    """Build the CNN input from the stored arrays.

    wp:     float [B, L]  win probability from the player's view (padded)
    side:   long  [B]     0 = player is White, 1 = player is Black
    length: long  [B]     number of real plies

    Returns x [B, 6, L] and mask [B, L]. Channels:
        0  win probability, centred to [-1, 1]
        1  change in win probability since the previous ply
        2  the PLAYER's own mistakes: win-probability drop on their moves, else 0
        3  the OPPONENT's mistakes: win-probability drop on their moves, else 0
        4  1 on plies played by the player, 0 on the opponent's plies
        5  padding mask (1 = real ply)
    Channels 2 and 3 let the network see both players' accuracy side by side.
    """
    _, seq_len = wp.shape
    t = torch.arange(seq_len, device=wp.device).unsqueeze(0)            # [1, L]
    mask = (t < length.unsqueeze(1)).to(wp.dtype)                        # [B, L]

    prev = torch.cat([torch.full_like(wp[:, :1], 0.5), wp[:, :-1]], dim=1)
    diff = wp - prev                                                     # player's view
    # Ply index t (0-based) is played by White when t is even, by Black when odd.
    own = ((t % 2) == side.unsqueeze(1)).to(wp.dtype) * mask
    opp = (1.0 - own) * mask

    centred = (wp - 0.5) * 2.0 * mask
    delta = diff * DELTA_SCALE * mask
    own_loss = (-diff).clamp(min=0.0) * DELTA_SCALE * own   # player moved, their eval fell
    opp_loss = diff.clamp(min=0.0) * DELTA_SCALE * opp      # opponent moved, their eval fell

    x = torch.stack([centred, delta, own_loss, opp_loss, own, mask], dim=1)
    return x, mask


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = dilation * (kernel - 1) // 2  # "same" length output
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class RatingCNN(nn.Module):
    """Dilated conv blocks + masked avg/max pooling, optionally joined by a
    feature MLP, then a small head.

    Outputs one value per sample: the STANDARDISED rating (train.py converts
    it back to Elo points).

    The feature normalisation (mean, std) and an on/off mask per feature are
    stored as buffers, so they travel inside the saved weights and serving
    can't use different statistics than training.
    """

    def __init__(self, channels=(32, 64, 64, 128), kernel=5,
                 dilations=(1, 2, 4, 8), dropout=0.2, hidden=64,
                 use_features=True, feat_hidden=64, n_feats=N_FEATURES):
        super().__init__()
        if len(channels) != len(dilations):
            raise ValueError("channels and dilations must have the same length")
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")

        blocks, in_ch = [], IN_CHANNELS
        for out_ch, dil in zip(channels, dilations):
            blocks.append(ConvBlock(in_ch, out_ch, kernel, dil, dropout))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        self.use_features = use_features
        self.register_buffer("feat_mean", torch.zeros(n_feats))
        self.register_buffer("feat_std", torch.ones(n_feats))
        self.register_buffer("feat_mask", torch.ones(n_feats))  # 0 = feature switched off
        head_in = 2 * in_ch
        if use_features:
            self.feat_net = nn.Sequential(
                nn.Linear(n_feats, feat_hidden), nn.ReLU(), nn.Dropout(dropout))
            head_in += feat_hidden

        self.head = nn.Sequential(
            nn.Linear(head_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def set_feature_stats(self, mean, std, mask=None):
        """Store the train-set feature statistics (and optional on/off mask)."""
        ref = self.feat_mean
        with torch.no_grad():
            self.feat_mean.copy_(torch.as_tensor(mean, dtype=ref.dtype, device=ref.device))
            std = torch.as_tensor(std, dtype=ref.dtype, device=ref.device)
            # A constant column (std ~ 0, e.g. "result_known" when every training
            # game has a result) would explode on any different value at serving
            # time, so use std = 1 there: it then contributes just (x - mean).
            self.feat_std.copy_(torch.where(std < 1e-6, torch.ones_like(std), std))
            if mask is not None:
                self.feat_mask.copy_(torch.as_tensor(mask, dtype=ref.dtype, device=ref.device))

    def forward(self, wp, side, length, feats=None):
        x, mask = make_inputs(wp, side, length)
        m = mask.unsqueeze(1)                                            # [B, 1, L]

        h = x
        for block in self.blocks:
            h = block(h) * m                                             # keep padding at 0

        avg = h.sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        mx = h.masked_fill(m == 0, -1e4).amax(dim=2)
        pooled = torch.cat([avg, mx], dim=1)

        if self.use_features:
            if feats is None:
                raise ValueError("this model was built with use_features=True; pass feats")
            f = (feats - self.feat_mean) / self.feat_std * self.feat_mask
            pooled = torch.cat([pooled, self.feat_net(f)], dim=1)

        return self.head(pooled).squeeze(-1)


def build_model(cfg):
    """Create the model from the `train` section of params.yaml."""
    return RatingCNN(
        channels=tuple(cfg["channels"]),
        kernel=int(cfg["kernel"]),
        dilations=tuple(cfg["dilations"]),
        dropout=float(cfg["dropout"]),
        hidden=int(cfg["hidden"]),
        use_features=bool(cfg.get("use_features", True)),
        feat_hidden=int(cfg.get("feat_hidden", 64)),
    )
# Speech Separation with Adaptive Slice Fusion (Direction A+C)

> **EEB588 Final Project** — Conv-TasNet + SC-CHM Fusion + Direction A (Input Scaling) + Direction C (Phase Shift)

---

## Overview

This project implements a **two-speaker speech separation** system based on **Conv-TasNet**, with architectural extensions developed iteratively during the course:

1. **Dual-Path Encoder**: Combines a 1D convolutional encoder and an STFT encoder, concatenating their outputs for richer time-frequency representation.
2. **SC-CHM Block** (Shuffled Convolution + Channel Harmonization Module): Replaces the standard depthwise conv block with a dual-path gated block using channel shuffling.
3. **Multi-Slice TCN** (`Slice=2`): Two parallel TCN branches, each accumulating skip connections independently.
4. **Direction A+C Adaptive Fusion**: Replaces the static einsum-based slice weighting with an input-dependent mechanism combining:
   - **Direction A** — Input Scaling: `Pooling → FC → Sigmoid → α` (per-weight retention ratio)
   - **Direction C** — Phase Shift: `Pooling → FC → Gumbel-Softmax → δ` (cyclic shift selection)

### Motivation

The original multi-slice design collapses to a single scalar weight per slice:

```python
# Before: all slices share the same weight sum — no adaptability
slic_into_weight = th.einsum('i,jkl->ijkl', [self.wList, skip_connection])
Slices_Output    = th.einsum('ijkl->jkl',   [slic_into_weight])
# result = (w₀+w₁+w₂+w₃) × skip — identical for every input
```

With Direction A+C, each slice gets a **different, input-dependent** effective weight:

```
w_eff(s) = Σ_{d=0}^{K-1}  δ_d · α_{(s+d)%K} · w_{(s+d)%K}
output   += w_eff(s) · h⁽ˢ⁾
```

---

## Architecture

### Encoder (Dual-Path)

```python
# 1D Conv path
w = self.encoder_1d(x)   # Conv1D(1, 512, L=16, stride=8) → (B, 512, T)
w = w[:, :256, :]        # take first 256 channels → (B, 256, T)

# STFT path
out = th.stft(x, n_fft=512, hop_length=8, win_length=64,
              return_complex=True, window=th.hann_window(64))
out = out.real           # take real part
out = out[:, :256, :-2]  # trim to match time dim → (B, 256, T)

# Concatenate
w = th.cat((w, out), 1)  # → (B, 512, T)
```

> **Note**: Unlike baseline Conv-TasNet, no ReLU is applied to the encoder output.

### SC-CHM Conv1D Block (Key Innovation)

Each Conv1D block runs **two parallel paths** that cross-multiply (gated harmonization):

```
Input x (B, B_ch=256, T)
    │
1×1 Conv → PReLU → gLN
    │
    ├──── ChannelShuffle → GroupConv ──→ tanh  (shufftan)
    │                               ──→ sigmoid (shuffsigm)
    │
    └──── DepthwiseConv → 1×1 Conv  ──→ sigmoid (depsigm)
                                    ──→ tanh    (deptan)
    
cross: _x_up   = shufftan  × depsigm
       _x_down = shuffsigm × deptan
       y = cat(_x_up, _x_down)   # (B, 512, T)
    │
PReLU → gLN
    ├── sconv (1×1) → residual (x = x + out)
    └── skip_out (1×1) → skip connection (B, Sc=256, T)
```

> **Channel Shuffle**: Uses `channel_shuffleforsound` — reshapes `(B, C/groups, groups, T)` then transposes dims 1,2.  
> `groups` is tied to dilation: `groups = dilation × (kernel_size - 1) // 2`, so it varies per block.

### Direction A+C Adaptive Controller

Implemented **inside the `forward()` loop**, computed after each slice's skip connections accumulate:

```python
# After Slice s finishes all R×X blocks:
c_skip = skip_connection.mean(dim=-1)              # (B, 256) — AvgPool over time

# Direction A: Input Scaling
alpha = th.sigmoid(self.fc_scale(c_skip))          # fc_scale: Linear(256, 4) → (B, 4)

# Direction C: Phase Shift
delta = F.gumbel_softmax(self.fc_shift(c_skip),
                          tau=self.tau,
                          hard=not self.training,
                          dim=-1)                  # fc_shift: Linear(256, 4) → (B, 4)

# A+C fusion: cyclic index shift
indices       = [(Slice + d) % K for d in range(K)]   # K=4
shifted_w     = self.wList[indices]                    # (K,)
shifted_alpha = alpha[:, indices]                      # (B, K)
scaled_shifted = shifted_alpha * shifted_w             # (B, K)
w_eff          = (delta * scaled_shifted).sum(dim=-1)  # (B,)

weighted_skip  = skip_connection * w_eff.unsqueeze(-1).unsqueeze(-1)
Slices_Output  = Slices_Output + weighted_skip
```

> **Key implementation detail**: Both `fc_scale` and `fc_shift` receive the **same** `c_skip` (mean-pooled from `skip_connection` of the current slice), **not** from the encoder output `w`.  
> `fc_shift` input dim is `N//2 = 256` (same as `Sc=256`), matching `c_skip` exactly.

---

## File Structure

```
denoiser/
├── model_SC_CHM_Fusion.py   # Main model: MS_SL2_split_model with SC-CHM + Direction A+C
├── model.py                 # Legacy variants (ConvTasNet, MB_ConvTasNet, etc.)
├── trainnew_blue.py         # Training entry point (loads best.pt.tar, fine-tunes with strict=False)
├── trainer.py               # SiSnrTrainer: SI-SNR + PIT, LR scheduling, Gumbel-τ annealing
├── conf.py                  # Hyperparameters and data paths
├── dataset.py               # Data loading: Kaldi .scp → ChunkSplitter → DataLoader
├── audio.py                 # WaveReader, WAV I/O
├── utils.py                 # Logger, JSON dump/load
├── train_blue.sh            # Shell script to launch training
├── best.pt.tar              # Pre-trained checkpoint used as fine-tune base (64 MB)
├── checkpoints/
│   ├── best.pt.tar          # Best model from training run (245 MB)
│   ├── trainer.log          # Full training log (50 epochs)
│   ├── trainer.json         # Trainer config snapshot
│   ├── mdl.json             # Model config snapshot
│   └── data.json            # Data config snapshot
├── loss_vs_epoch.png
├── loss_vs_epoch_comparison.png
├── loss_vs_epoch_e50.png
└── md/                      # Design documents and implementation plans
```

---

## Model Configuration (`conf.py`)

```python
nnet_conf = {
    "L": 16,          # Encoder filter length (samples)
    "N": 512,         # Encoder output channels
    "X": 8,           # Conv1D blocks per repeat
    "R": 2,           # Number of repeats per slice
    "B": 256,         # Bottleneck channels
    "Sc": 256,        # Skip-connection channels
    # "Slice": 2,     # commented out → uses model default = 2
    "H": 512,         # Hidden channels in Conv1D blocks
    "P": 3,           # Kernel size
    "norm": "gLN",    # Global Layer Normalization
    "num_spks": 2,    # Two-source separation
    "non_linear": "sigmoid"
}

chunk_len  = 3        # seconds
chunk_size = 48000    # = 3 × 16000 samples
```

| Component | Parameters |
|-----------|-----------|
| Total model | ~20.64M |
| `fc_scale` (Linear 256→4) | 1,028 |
| `fc_shift` (Linear 256→4) | 1,028 |
| **Direction A+C total** | **2,056** (<3K) |

---

## Training

### Loss Function

**SI-SNR** (Scale-Invariant SNR) with **PIT** (Permutation Invariant Training):

```
s_target = (<x_zm, s_zm> / ‖s_zm‖²) × s_zm
SI-SNR   = 20 × log₁₀(‖s_target‖ / ‖x_zm − s_target‖)
Loss     = −mean(max_permutation SI-SNR)
```

### Optimizer

| Setting | Value |
|---------|-------|
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-8) |
| Epochs | 50 |
| Batch size | 2 |
| Training subset | 50% of training data (`subset_ratio=0.5`) |

### Gumbel-Softmax Temperature Annealing

Implemented in `trainer.py`:

```python
if hasattr(self.nnet, 'tau'):
    self.nnet.tau = max(0.1, 1.0 - (self.cur_epoch / num_epochs) * 0.9)
```

| Epoch | τ | Behavior |
|-------|---|----------|
| 1 | 1.00 | Soft — explores all shift indices |
| 25 | 0.55 | Moderate focus |
| 50 | 0.10 | Near hard-argmax |
| Inference | hard=True | Exact one-hot argmax |

### Fine-tuning Strategy

`trainnew_blue.py` loads `best.pt.tar` with `strict=False`, allowing pre-trained weights to carry over while the new `fc_scale` and `fc_shift` layers are randomly initialized:

```python
old_model = th.load('best.pt.tar', map_location=device)
nnet.load_state_dict(old_model['model_state_dict'], strict=False)
```

---

## Training Results

Training ran for **50 epochs** (2026-05-26 → 2026-06-03):

| Metric | Value |
|--------|-------|
| Final Train SI-SNR (epoch 50) | **14.85 dB** |
| Final Dev SI-SNR (epoch 50) | **13.39 dB** |
| Final LR | 3.125e-05 |
| Train time / epoch | ~52.9 min |
| Dev time / epoch | ~5.1 min |

> Loss is reported as negative SI-SNR (minimization objective).  
> A final loss of −14.85 means SI-SNR ≈ **14.85 dB**.

Loss curves:

- [`loss_vs_epoch.png`](loss_vs_epoch.png)
- [`loss_vs_epoch_comparison.png`](loss_vs_epoch_comparison.png)
- [`loss_vs_epoch_e50.png`](loss_vs_epoch_e50.png)

---

## How to Run

### Requirements

```bash
pip install torch torchaudio scipy numpy
```

### Training

```bash
python trainnew_blue.py \
    --gpus 0 \
    --epochs 50 \
    --checkpoint ./checkpoints \
    --batch-size 2 \
    --num-workers 0 \
    --trainer_type origin
```

**`--trainer_type` options:**

| Mode | Behavior |
|------|----------|
| `origin` | Train all parameters |
| `repeat` | Freeze all except repeat index `[2]=='1'` (second repeat fine-tune) |
| `add_block` | Freeze all except block index `[3]=='8'` (9th block fine-tune) |

### Data Format (Kaldi-style)

```
tr/
  mix.scp    # key → mixed audio path
  spk1.scp   # key → speaker 1 audio path
  spk2.scp   # key → speaker 2 audio path
cv/
  (same structure for validation)
```

---

## References

- Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation. *IEEE/ACM TASLP*.
- Zhang, X., et al. (2018). ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices. *CVPR*.
- Jang, E., Gu, S., & Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

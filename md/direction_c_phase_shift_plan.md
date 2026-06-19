# Direction C: Phase Shift — Detailed Implementation Plan

> [!NOTE]
> **Status (2026-06-19):** This plan has been **implemented** in [model_SC_CHM_Fusion.py](file:///c:/Users/Wei/Downloads/denoiser/model_SC_CHM_Fusion.py). The sections below have been updated to reflect the actual code. Key differences from the original proposal: (1) no separate `PhaseShiftPredictor` class — uses an inline `nn.Linear` + `F.gumbel_softmax`; (2) no `_get_shifted_weight` method — fusion logic is inline; (3) input is `skip_connection.mean(dim=-1)` (pooled skip connections, 256-dim), not the bottleneck `y`; (4) Direction C is combined with Direction A (Input Scaling) in a single fused formula; (5) the predictor runs **inside** the slice loop, after each slice finishes.

## 1. Problem Statement: Why Phase Shift?

### 1.1 Current Behavior (What's Broken)

Looking at the current weight fusion in [model_SC_CHM_Fusion.py](file:///c:/Users/Wei/Downloads/denoiser/model_SC_CHM_Fusion.py) (lines 592–606):

```python
# Current: For EVERY slice, ALL 4 weights are used identically
slic_into_weight = th.einsum('i,jkl->ijkl', [self.wList, skip_connection])
#   wList:          (K=4,)
#   skip_connection: (B, Sc, T)
#   result:          (4, B, Sc, T)  — broadcasts each w_i across the spatial dims

Slices_Output = th.einsum('ijkl->jkl', [slic_into_weight])
#   Sums over the K=4 dimension → (B, Sc, T)
#   This equals: (w₀ + w₁ + w₂ + w₃) × skip_connection
```

> [!CAUTION]
> **The 4 weights are degenerate.** Every slice's skip connection is multiplied by the **same scalar** `Σwᵢ`. Having 4 separate weights provides no benefit over a single scalar — they always sum to one value. The cyclic mapping `i* = (i mod K)+1` is conceptually present but **not functionally realized**.

### 1.2 What Phase Shift Fixes

Phase Shift assigns **one specific weight** to each slice, following a cyclic pattern whose starting point (Δ) adapts to the input:

```
Fixed cyclic (current spirit):
  Slice 0 → w[0],  Slice 1 → w[1]

Phase Shift with Δ=0:  Slice 0 → w[0],  Slice 1 → w[1]
Phase Shift with Δ=1:  Slice 0 → w[1],  Slice 1 → w[2]
Phase Shift with Δ=2:  Slice 0 → w[2],  Slice 1 → w[3]
Phase Shift with Δ=3:  Slice 0 → w[3],  Slice 1 → w[0]
```

**Analogy:** Imagine a combination lock with 4 numbers. The current code always uses all 4 at once (meaningless). Phase Shift rotates the dial to a different starting position based on the input, and each slice reads one number from the sequence.

---

## 2. Mathematical Formulation

### 2.1 Original Cyclic Mapping

$$i^* = (i \bmod K) + 1$$

Where:
- `i` = slice index (0, 1, ..., S-1)
- `K` = number of weights (4)
- `i*` = which weight index to use for slice `i`

### 2.2 Phase-Shifted Cyclic Mapping

$$i^* = ((i + \Delta) \bmod K) + 1$$

Where:
- `Δ ∈ {0, 1, ..., K-1}` = phase shift amount
- `Δ` is **input-dependent**: determined by a small predictor network
- Made **differentiable** via Gumbel-Softmax

### 2.3 Differentiable Selection

Since `Δ` is a discrete choice, we use Gumbel-Softmax to make it differentiable:

```
logits = FC(GlobalAvgPool(y))           # (B, K) — raw scores for each shift
δ = GumbelSoftmax(logits, τ)            # (B, K) — soft one-hot during training

For slice s, the effective weight:
  w_eff(s) = Σ_d  δ_d · w[(s+d) mod K]   # weighted average of possible shifts
```

- **Training** (τ > 0): soft interpolation — gradients flow through all paths
- **Inference** (hard=True): argmax — picks one discrete Δ

---

## 3. Architecture of Changes

### 3.1 Overview Diagram

```
                    Input Waveform (B, S)
                           │
                           ▼
                    ┌──────────────┐
                    │   Encoder    │
                    └──────┬───────┘
                           │ (B, 512, T)
                           ▼
                    ┌──────────────┐
                    │  cLN + Proj  │
                    └──────┬───────┘
                           │ y: (B, 256, T)
                           ▼
               ┌───────────────────────┐
               │ for Slice in slices:  │
               │  ┌──────────────────┐ │
               │  │  TCN Blocks      │ │
               │  │  (R×X Conv1D)    │ │
               │  └────────┬─────────┘ │
               │           │ skip_connection: (B, 256, T)
               │           ▼           │
               │  ┌──────────────────┐ │
               │  │ c_skip = skip    │ │
               │  │  .mean(dim=-1)   │ │  ◄── pool over time
               │  └────────┬─────────┘ │
               │           │ (B, 256)   │
               │     ┌─────┴─────┐     │
               │     ▼           ▼     │
               │  ┌────────┐ ┌──────┐  │
               │  │fc_scale│ │fc_sh.│  │  ◄── Direction A + C
               │  │→sigmoid│ │→Gumb.│  │
               │  └───┬────┘ └──┬───┘  │
               │      │α (B,4)  │δ(B,4)│
               │      ▼         ▼      │
               │  ┌──────────────────┐  │
               │  │ A+C Fusion:      │  │
               │  │ shifted_w, α_sh  │  │
               │  │ w_eff = Σ δ·α·w  │  │
               │  └────────┬─────────┘  │
               │           │ w_eff (B,) │
               │           ▼            │
               │  Slices_Output +=      │
               │   skip * w_eff         │
               └───────────┬────────────┘
                           │ (B, Sc, T)
                           ▼
                    ┌──────────────┐
                    │ PReLU → Mask │
                    │ → Decode     │
                    └──────────────┘
```

### 3.2 What Changes vs. What Stays

| Component | Status | Details |
|-----------|--------|---------|
| Encoder (1D Conv + STFT) | **Unchanged** | |
| cLN + 1×1 Proj | **Unchanged** | |
| TCN Slices (Conv1DBlocks) | **Unchanged** | |
| `wList` (4 learnable weights) | **Kept** | Still K=4 base weights |
| `einsum` fusion logic | **Replaced** | New A+C fused per-slice weight selection |
| `fc_scale` (Direction A) | **New** | `nn.Linear(Sc, 4)` — 1,028 params |
| `fc_shift` (Direction C) | **New** | `nn.Linear(N//2, 4)` — 1,028 params |
| Mask + Decoder | **Unchanged** | |

---

## 4. Detailed Code Changes

### 4.1 Inline Phase Shift (No Separate Class)

> [!IMPORTANT]
> The actual implementation does **not** use a separate `PhaseShiftPredictor` class or a `_get_shifted_weight` method. Instead, everything is inline inside `MS_SL2_split_model.__init__()` and `forward()`.

**No `nn.AdaptiveAvgPool1d`** — the code uses `.mean(dim=-1)` on `skip_connection` to pool over the time dimension.

**No separate method** — the cyclic index lookup + weighted sum is computed inline in the slice loop.

### 4.2 Modifications to `__init__`

**Location:** Inside `MS_SL2_split_model.__init__` (around line 446–457)

```python
# wList stays the same
self.wList = nn.Parameter((max-min)*th.rand(4)+min, requires_grad=True)

# Direction A: Input Scaling — FC(Sc→4)
self.fc_scale = nn.Linear(Sc, 4)           # 256×4+4 = 1,028 params

# Direction C: Phase Shift — FC(N//2→4)
self.fc_shift = nn.Linear(N // 2, 4)       # 256×4+4 = 1,028 params
self.tau = 1.0  # Gumbel-Softmax temperature (anneal during training)
```

> [!NOTE]
> Both `fc_scale` (Direction A) and `fc_shift` (Direction C) take the **same** input `c_skip` (pooled skip connections) — they are defined as separate linear layers but share the same input source.

### 4.3 Modifications to `forward`

**Location:** The slice loop in `forward()` (lines ~562–603 in actual code).

#### Before (original `einsum` code):
```python
# All 4 weights applied identically to every slice
slic_into_weight = th.einsum('i,jkl->ijkl', [self.wList, skip_connection])
Slices_Output = th.einsum('ijkl->jkl', [slic_into_weight])
```

#### After (actual A+C fused implementation):
```python
K = self.wList.shape[0]  # K=4

skip_connection = 0
Slice_input = y
Slices_Output = 0
for Slice in range(self.slice):
    for i in range(self.R):
        for j in range(self.X):
            skip, y = self.slices[Slice][i][j](y)
            skip_connection = skip_connection + skip

    # ── Direction A + C: computed INSIDE the loop, AFTER TCN blocks ──
    c_skip = skip_connection.mean(dim=-1)              # (B, Sc=256)
    alpha = th.sigmoid(self.fc_scale(c_skip))          # (B, 4)  — Direction A
    delta = F.gumbel_softmax(self.fc_shift(c_skip),
                              tau=self.tau,
                              hard=not self.training,
                              dim=-1)                  # (B, 4)  — Direction C

    # A+C fusion: w_eff(s) = Σ_d δ_d · α[(s+d)%K] · w[(s+d)%K]
    indices = [(Slice + d) % K for d in range(K)]      # cyclic indices
    shifted_w = self.wList[indices]                     # (K,)  phase-shifted base weights
    shifted_alpha = alpha[:, indices]                   # (B, K) phase-shifted scaling factors
    scaled_shifted = shifted_alpha * shifted_w          # (B, K) = αₖ · wₖ (shifted)
    w_eff = (delta * scaled_shifted).sum(dim=-1)        # (B,)

    weighted_skip = skip_connection * w_eff.unsqueeze(-1).unsqueeze(-1)  # (B, Sc, T)
    Slices_Output = Slices_Output + weighted_skip

    skip_connection = 0
    y = Slice_input
```

> [!IMPORTANT]
> Key differences from original plan:
> - **No `phase_predictor` call before the loop** — `fc_shift` and `fc_scale` are called **inside** the loop, after each slice's TCN blocks finish.
> - **Input is `skip_connection.mean(dim=-1)`** (accumulated skip connections pooled over time), NOT the bottleneck `y`.
> - **Direction A (`alpha`) and Direction C (`delta`) are combined** into a single `w_eff` per slice.

---

## 5. Tensor Shape Walkthrough

Let's trace through with concrete shapes (`B=4` batch, `K=4` weights, `S=2` slices, `Sc=256`, `T=1249`):

```
Slice 0 Processing
──────────────────
skip_connection:  (4, 256, 1249)    ← sum of all skip outputs from slice 0's TCN blocks

c_skip = skip_connection.mean(dim=-1)   ← (4, 256)  pool over time

Direction A (Input Scaling):
  fc_scale(c_skip):  (4, 256) → Linear(256,4) → (4, 4)
  alpha = sigmoid(…):  (4, 4)   ← per-weight scaling factors ∈ (0,1)

Direction C (Phase Shift):
  fc_shift(c_skip):  (4, 256) → Linear(256,4) → (4, 4)  ← logits
  delta = GumbelSoftmax(…):  (4, 4)
    Training:   soft one-hot, e.g. [[0.3, 0.1, 0.5, 0.1], ...]
    Inference:  hard one-hot, e.g. [[0, 0, 1, 0], ...]

A+C Fusion (Slice=0):
  indices = [0, 1, 2, 3]             ← (0+0)%4, (0+1)%4, (0+2)%4, (0+3)%4
  shifted_w = wList[[0,1,2,3]]       ← (4,)  = [w₀, w₁, w₂, w₃]
  shifted_alpha = alpha[:, [0,1,2,3]] ← (4, 4) = [α₀, α₁, α₂, α₃]
  scaled_shifted = shifted_alpha * shifted_w  ← (4, 4) element-wise
  w_eff = (delta * scaled_shifted).sum(dim=-1)  ← (4,)

  Example (batch=0), delta ≈ [0, 0, 1, 0] (Δ=2):
    w_eff[0] ≈ 0·(α₀·w₀) + 0·(α₁·w₁) + 1·(α₂·w₂) + 0·(α₃·w₃)
             = α₂ · w₂

weighted_skip:  (4, 256, 1249)  ← skip_connection * w_eff[:, None, None]

Slice 1 Processing
──────────────────
skip_connection:  (4, 256, 1249)  ← FRESH accumulation from slice 1's TCN blocks

c_skip = skip_connection.mean(dim=-1)   ← (4, 256)  (different values from slice 0!)
alpha = sigmoid(fc_scale(c_skip))       ← (4, 4)    (recomputed for this slice)
delta = GumbelSoftmax(fc_shift(c_skip)) ← (4, 4)    (recomputed for this slice)

A+C Fusion (Slice=1):
  indices = [1, 2, 3, 0]             ← (1+0)%4, (1+1)%4, (1+2)%4, (1+3)%4
  shifted_w = wList[[1,2,3,0]]       ← [w₁, w₂, w₃, w₀]
  shifted_alpha = alpha[:, [1,2,3,0]] ← [α₁, α₂, α₃, α₀]
  w_eff = (delta * scaled_shifted).sum(dim=-1)

  Example (batch=0), delta ≈ [0, 1, 0, 0] (Δ=1):
    w_eff[0] ≈ 0·(α₁·w₁) + 1·(α₂·w₂) + 0·(α₃·w₃) + 0·(α₀·w₀)
             = α₂ · w₂

Result
──────
Slices_Output: (4, 256, 1249)   ← weighted sum of both slices
```

> [!TIP]
> Key improvements: (1) each slice gets a **different** effective weight via cyclic shifting; (2) `alpha` adds per-weight scaling on top; (3) both `alpha` and `delta` are **recomputed per slice** from that slice's own skip connections, so the model adapts to each slice's features independently.

---

## 6. Gumbel-Softmax Details

### 6.1 What It Does

The Gumbel-Softmax trick allows sampling from a categorical distribution while keeping gradients flowing:

```python
# Standard: argmax is not differentiable
Δ = argmax(logits)  # ✗ no gradient

# Gumbel-Softmax: differentiable relaxation
g_i ~ Gumbel(0, 1)                          # sample noise
δ_i = exp((logits_i + g_i) / τ)             # apply temperature
δ_i = δ_i / Σ_j exp((logits_j + g_j) / τ)  # normalize (softmax)
```

### 6.2 Temperature τ Schedule

| Phase | τ Value | Behavior |
|-------|---------|----------|
| Early training | 1.0 – 2.0 | Soft (explores all shifts equally) |
| Mid training | 0.5 – 1.0 | Partially focused |
| Late training | 0.1 – 0.5 | Nearly discrete (commits to a shift) |
| Inference | hard=True | Exact argmax (straight-through estimator) |

**Implementation option:** Anneal τ linearly per epoch:
```python
# In trainer, before each epoch:
model.tau = max(0.1, 1.0 - epoch * (0.9 / total_epochs))
```

### 6.3 Straight-Through Estimator (hard=True)

During inference:
- **Forward pass:** uses `argmax` (discrete one-hot)
- **Backward pass:** uses soft Gumbel-Softmax gradients (only relevant if fine-tuning)

PyTorch's `F.gumbel_softmax(hard=True)` handles this automatically.

---

## 7. Files Modified

### 7.1 `model_SC_CHM_Fusion.py` — 2 changes

| Change | Location | Description |
|--------|----------|-------------|
| Add `self.fc_scale`, `self.fc_shift`, `self.tau` to `__init__` | Lines 452–457 | 3 new lines (Direction A + C) |
| Replace `einsum` fusion in `forward` slice loop | Lines 562–603 | Inline A+C fusion (~20 lines) |

> [!NOTE]
> No separate `PhaseShiftPredictor` class was created. No `_get_shifted_weight` method was created. All logic is inline.

### 7.2 `trainer.py` — 1 optional change

| Change | Location | Description |
|--------|----------|-------------|
| Add τ annealing | Inside `run()` loop, before `self.train()` | 1 line per epoch |

### 7.3 `conf.py` — No changes required

`fc_scale` and `fc_shift` derive their dimensions from existing constructor args (`Sc`, `N`).

---

## 8. Backward Compatibility

### 8.1 Loading Old Checkpoints

Since `PhaseShiftPredictor` is a **new** module, loading old `best.pt.tar` with `strict=False` will:
- ✅ Load all existing weights (encoder, slices, wList, decoder, etc.)
- ✅ Initialize `phase_predictor` randomly (it's new)
- ✅ No conflict with existing parameter names

### 8.2 Standalone Test

The `SL2_split()` function at the bottom of the file should still work. The model will use random phase predictions initially.

---

## 9. Summary of Parameter Overhead

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Existing `wList` | 4 | Kept as-is |
| `fc_scale` (Direction A) | 256 × 4 + 4 = 1,028 | `nn.Linear(Sc, 4)` |
| `fc_shift` (Direction C) | 256 × 4 + 4 = 1,028 | `nn.Linear(N//2, 4)` |
| `.mean(dim=-1)` pooling | 0 | No parameters (replaces `AdaptiveAvgPool1d`) |
| **Total new** | **2,056** | ~0.002M, negligible |

For context, the full model has ~2-5M parameters. This adds **~0.04–0.1%** overhead.

---

## 10. Expected Behavioral Difference

```
┌──────────────────────────────────────────────────────────┐
│                     CURRENT MODEL                        │
│                                                          │
│  BUS noise input:   Slice 0 → (w₀+w₁+w₂+w₃)·h⁽⁰⁾     │
│                     Slice 1 → (w₀+w₁+w₂+w₃)·h⁽¹⁾     │
│                                                          │
│  CAFE noise input:  Slice 0 → (w₀+w₁+w₂+w₃)·h⁽⁰⁾     │  ← SAME weights!
│                     Slice 1 → (w₀+w₁+w₂+w₃)·h⁽¹⁾     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                  PHASE SHIFT MODEL                       │
│                                                          │
│  BUS noise input:   Δ=0 → Slice 0 → w₀·h⁽⁰⁾           │
│    (predictor)            Slice 1 → w₁·h⁽¹⁾           │
│                                                          │
│  CAFE noise input:  Δ=2 → Slice 0 → w₂·h⁽⁰⁾           │  ← DIFFERENT!
│    (predictor)            Slice 1 → w₃·h⁽¹⁾           │
└──────────────────────────────────────────────────────────┘
```

The model can now **adapt its weight assignment** based on input characteristics, while preserving the cyclic structure (`i+1` always follows `i` in the rotation).

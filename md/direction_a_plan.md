# 方向 A+C：Input Scaling + Phase Shift — 實作計畫

> [!IMPORTANT]
> **狀態：已更新（2026-06-19）** — 本文件已校正為與 `model_SC_CHM_Fusion.py` 的實際實作一致。
> 主要差異：fc_scale 使用 `nn.Linear(Sc=256, 4)`；輸入來自 `skip_connection.mean(dim=-1)`（每個 Slice 內）；
> 方向 A（alpha）與方向 C（delta / Gumbel-Softmax）融合為 A+C。

> [!NOTE]
> 此文件原始版本根據投影片「方向 A 的程式改動」與「c = h[0].mean(dim=-1) 拆解」的規格撰寫，
> 現已更新以反映 `model_SC_CHM_Fusion.py` 中的實際程式碼。

---

## 1. 核心概念

| 項目 | 說明 |
|------|------|
| **目的** | 輸入自適應的增益控制 |
| **保留** | w₁~w₄ 作為 base weight（原有 cyclic weight） |
| **新增** | scaling factor α（方向 A）+ phase shift δ（方向 C），由 skip_connection pooling + FC 產生 |
| **公式** | w_eff⁽ˢ⁾ = Σ_d δ_d · α[(s+d)%K] · w[(s+d)%K] |
| **類比** | 固定 EQ → 自動 EQ + 自動相位偏移 |
| **額外參數** | fc_scale: 256×4+4 = 1,028 + fc_shift: 256×4+4 = 1,028 = **共 2,056** |

### 1.1 完整流程

```
Pooling → FC → Sigmoid → Scaling
```

1. **Pooling**：從特徵中提取全局表徵，壓掉時間維度 T
2. **FC**：線性映射到 K=4 個值
3. **Sigmoid**：限制 α ∈ (0, 1)，作為每個 weight 的縮放因子
4. **Scaling**：`w_scaled = α · w_base`，然後用 scaled weight 做 cyclic weighting

---

## 2. 實際實作（對應 `model_SC_CHM_Fusion.py`）

### 2.1 `__init__()` 新增定義

```python
# 原有的程式（不動）
self.slices = self._build_slices(...)          # TCN slices
self.wList = nn.Parameter(...)                 # (K=4,) cyclic base weights

# ★ 方向 A: Input Scaling (1 行)
self.fc_scale = nn.Linear(Sc, 4)   # Sc=256, 額外參數: 256×4+4 = 1,028

# ★ 方向 C: Phase Shift (1 行 + tau)
self.fc_shift = nn.Linear(N // 2, 4)   # D=256, 額外參數: 256×4+4 = 1,028
self.tau = 1.0  # Gumbel-Softmax 溫度（在訓練中退火）
```

### 2.2 `forward()` — 在每個 Slice 的 TCN blocks 結束後（inline，無獨立 class/method）

```python
for Slice in range(self.slice):
    # --- 原有：跑這個 Slice 的 TCN blocks ---
    for i in range(self.R):
        for j in range(self.X):
            skip, y = self.slices[Slice][i][j](y)
            skip_connection = skip_connection + skip

    # ★ 方向 A: Pooling + FC + Sigmoid（每個 Slice 各算一次）
    c_skip = skip_connection.mean(dim=-1)              # (B, Sc=256)
    alpha = th.sigmoid(self.fc_scale(c_skip))          # (B, 4)

    # ★ 方向 C: Gumbel-Softmax Phase Shift
    delta = F.gumbel_softmax(self.fc_shift(c_skip),
                              tau=self.tau, hard=not self.training, dim=-1)  # (B, 4)

    # ★ A+C 融合: w_eff(s) = Σ_d δ_d · α[(s+d)%K] · w[(s+d)%K]
    indices = [(Slice + d) % K for d in range(K)]      # 循環索引
    shifted_w = self.wList[indices]                      # (K,)
    shifted_alpha = alpha[:, indices]                    # (B, K)
    scaled_shifted = shifted_alpha * shifted_w           # (B, K)
    w_eff = (delta * scaled_shifted).sum(dim=-1)         # (B,)
    weighted_skip = skip_connection * w_eff.unsqueeze(-1).unsqueeze(-1)  # (B, Sc, T)
    Slices_Output = Slices_Output + weighted_skip

    skip_connection = 0
    y = Slice_input
```

### 2.3 改動量統計

| 位置 | 改動 |
|------|------|
| `__init__()` | 新增 `self.fc_scale = nn.Linear(Sc, 4)` | +1 行 |
| `__init__()` | 新增 `self.fc_shift = nn.Linear(N // 2, 4)` + `self.tau = 1.0` | +2 行 |
| `forward()` | 新增 A+C 融合邏輯（pooling → alpha → delta → w_eff → weighted_skip） | +~12 行 |
| **總計** | **~15 行新增，原有 TCN 結構不動** |

---

## 3. Pooling 來源：`c = h[0].mean(dim=-1)` 拆解

### 3.1 h 是什麼？

```
h = [slice(x) for slice in self.tcn_slices]
```

- h 是 Python list，裡面有 S 個 tensor（S = slice 數量）
- `h[0]` = 只取第 1 條 Slice 的輸出（index 0）

```
h[0]          h[1]          h[2]
Slice 1 輸出   Slice 2 輸出   Slice 3 輸出
(batch, 512, T)  (batch, 512, T)  (batch, 512, T)
```

### 3.2 h[0] 的三個維度

| dim | 意義 | 說明 |
|-----|------|------|
| dim=0 | batch | 幾句話一起算 |
| dim=1 | 512 channels | 各頻率成分 |
| dim=2 (= dim=-1) | T 個時間步 | ← mean 壓掉這一維 |

### 3.3 結果

```
h[0]:           (batch, 512, T)
                        ↓ .mean(dim=-1)
c:              (batch, 512)      ← T 那維被壓掉了
```

### 3.4 其他做法（ablation 實驗方向）

| 做法 | 程式碼 | 說明 |
|------|--------|------|
| **只用 Slice 1** ★ | `h[0].mean(-1)` | 投影片推薦的主要做法 |
| 平均所有 Slice | `stack(h).mean(0,-1)` | 融合所有 slice 資訊 |
| 用 TCN 前的輸入 | `x.mean(-1)` | 不依賴任何 slice 輸出 |

---

## 4. 對應 SC-CHM Fusion 模型的實作

> [!IMPORTANT]
> 以下為 `model_SC_CHM_Fusion.py` 的**實際實作**方式，非投影片的通用記號。

### 4.1 架構對應

| 概念 | SC-CHM 模型對應 | 維度 |
|------|----------------|------|
| Pooling 來源 | `skip_connection.mean(dim=-1)`（當前 Slice 的累積 skip） | (B, **Sc=256**) |
| FC (方向 A) | `self.fc_scale = nn.Linear(Sc, 4)`，Sc=256 | 256 → 4 |
| FC (方向 C) | `self.fc_shift = nn.Linear(N//2, 4)`，N//2=256 | 256 → 4 |
| Base weights | `self.wList = nn.Parameter(th.rand(4))`，K=4 | (4,) |

### 4.2 實際 Pooling 方案（唯一方案）

| 項目 | 說明 |
|------|------|
| **來源** | `skip_connection`（當前 Slice 的 TCN blocks 累積的 skip connections） |
| **Pooling** | `skip_connection.mean(dim=-1)` → (B, Sc=256) |
| **時機** | 每個 Slice 的 TCN blocks 跑完後，在 slice loop 內計算 |
| **特性** | alpha 和 delta 都是 **per-slice** 計算，每個 Slice 有各自的 alpha/delta |

> [!TIP]
> 每個 Slice 用自己累積的 `skip_connection` 做 pooling，意味著不同 Slice 會產生不同的 alpha 和 delta。
> 這讓模型能對每個 Slice 做不同程度的自適應調整。

### 4.3 實際程式碼（`model_SC_CHM_Fusion.py` L585–L600）

```python
# 在每個 Slice 的 TCN blocks 結束後（inline in forward()）：
c_skip = skip_connection.mean(dim=-1)              # (B, Sc=256)
alpha = th.sigmoid(self.fc_scale(c_skip))          # (B, 4)

delta = F.gumbel_softmax(self.fc_shift(c_skip),
                          tau=self.tau, hard=not self.training, dim=-1)  # (B, 4)

# A+C 融合
indices = [(Slice + d) % K for d in range(K)]      # 循環索引
shifted_w = self.wList[indices]                      # (K,)
shifted_alpha = alpha[:, indices]                    # (B, K)
scaled_shifted = shifted_alpha * shifted_w           # (B, K)
w_eff = (delta * scaled_shifted).sum(dim=-1)         # (B,)
weighted_skip = skip_connection * w_eff.unsqueeze(-1).unsqueeze(-1)  # (B, Sc, T)
Slices_Output = Slices_Output + weighted_skip
```

---

## 5. Sigmoid 的選擇理由

| 激活函數 | 範圍 | 行為 |
|---------|------|------|
| **Sigmoid(x)** ★ | **(0, 1)** | **投影片指定** — 純粹做「衰減/保留」，不放大 |
| Softplus(x) | (0, +∞) | 允許放大，但可能數值不穩定 |
| Sigmoid(x) × 2 | (0, 2) | 折衷方案，允許微幅放大 |

> [!NOTE]
> 投影片明確指定使用 **Sigmoid**。α ∈ (0, 1) 的語義是「每個 base weight 要保留多少比例」：
> - α ≈ 1.0 → 完全保留該 weight
> - α ≈ 0.0 → 幾乎抑制該 weight
> - α ≈ 0.5 → 保留一半
>
> 這與「自動 EQ」的類比一致：EQ 是調整各頻段的增益，Sigmoid 讓模型學會「該保留哪些 weight、抑制哪些 weight」。

---

## 6. 張量形狀追蹤

以 `B=4, K=4, Sc=256, T=1249` 為例：

```
步驟 1: Pooling（每個 Slice 內）
─────────────────────────────────
skip_connection:           (4, 256, 1249)  ← 當前 Slice 累積的 skip
c_skip = skip_connection.mean(dim=-1):  (4, 256)  ← 時間維度被壓掉

步驟 2: 方向 A — FC + Sigmoid
─────────────────────────────
fc_scale(c_skip):  (4, 4)   ← 線性映射 256 → 4
alpha:             (4, 4)   ← Sigmoid 後，例如 [[0.8, 0.3, 0.9, 0.5], ...]

步驟 3: 方向 C — Gumbel-Softmax
────────────────────────────────
fc_shift(c_skip):  (4, 4)   ← 線性映射 256 → 4
delta:             (4, 4)   ← Gumbel-Softmax 後，例如 [[0, 0, 1, 0], ...]
                              （training 時 soft，eval 時 hard one-hot）

步驟 4: A+C 融合（以 Slice=0 為例）
───────────────────────────────────
indices = [0, 1, 2, 3]               ← (Slice + d) % K, d=0..3
shifted_w = wList[[0,1,2,3]]          → (4,)  base weights
shifted_alpha = alpha[:, [0,1,2,3]]   → (4, 4)  scaling factors
scaled_shifted = shifted_alpha * shifted_w  → (4, 4)  = αₖ · wₖ
w_eff = (delta * scaled_shifted).sum(dim=-1)  → (4,)  每個 batch 一個有效 weight

步驟 5: Weighted Skip
─────────────────────
weighted_skip = skip_connection * w_eff[:, None, None]  → (4, 256, 1249)
Slices_Output += weighted_skip

以 Slice=1 為例:
indices = [1, 2, 3, 0]  ← 循環偏移！
→ 不同 Slice 會從不同的起始位置取 wList 和 alpha，實現 phase shift
```

---

## 7. 行為對比

```
┌────────────────────────────────────────────────────────┐
│                    現有模型                              │
│                                                        │
│  BUS 噪音:   Slice 0 → (w₀+w₁+w₂+w₃)·h⁽⁰⁾          │
│              Slice 1 → (w₀+w₁+w₂+w₃)·h⁽¹⁾          │
│  CAFE 噪音:  完全相同 ← 無法自適應!                     │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│               方向 A 模型（Input Scaling）               │
│                                                        │
│  BUS 噪音:  α=[0.9, 0.3, -, -]                        │
│    → Slice 0: 0.9·w₀·h⁽⁰⁾  (保留 90%)                │
│    → Slice 1: 0.3·w₁·h⁽¹⁾  (只保留 30%)              │
│                                                        │
│  CAFE 噪音: α=[0.4, 0.8, -, -]                        │
│    → Slice 0: 0.4·w₀·h⁽⁰⁾  (只保留 40%) ← 不同!      │
│    → Slice 1: 0.8·w₁·h⁽¹⁾  (保留 80%)  ← 不同!      │
└────────────────────────────────────────────────────────┘
```

---

## 8. 修改檔案清單

| 檔案 | 修改 | 行數 |
|------|------|------|
| `model_SC_CHM_Fusion.py` | `__init__` 新增 `self.fc_scale = nn.Linear(Sc, 4)` (Sc=256) | +1 行 |
| `model_SC_CHM_Fusion.py` | `__init__` 新增 `self.fc_shift = nn.Linear(N//2, 4)` + `self.tau = 1.0` | +2 行 |
| `model_SC_CHM_Fusion.py` | `forward` 新增 A+C 融合邏輯（pooling → alpha → delta → indices → w_eff → weighted_skip） | +~12 行 |
| `conf.py` | **不需修改** | 0 |
| `trainer.py` | 需加入 τ 退火邏輯（方向 C 的 `self.tau` 需逐 epoch 遞減） | ~數行 |

### 參數量統計

| 模組 | 公式 | 參數量 |
|------|------|--------|
| `fc_scale = nn.Linear(256, 4)` | 256×4 + 4 | **1,028** |
| `fc_shift = nn.Linear(256, 4)` | 256×4 + 4 | **1,028** |
| **A+C 總額外參數** | | **2,056** |

# 方向 A+C：Phase Shift + Input Scaling 實作計畫

> [!IMPORTANT]
> **✅ 已實作完成** — 此計畫已實作於 [model_SC_CHM_Fusion.py](file:///c:/Users/Wei/Downloads/denoiser/model_SC_CHM_Fusion.py)。
> 實作與原始計畫有以下差異：
> - **沒有獨立的 `AdaptiveController` 類別** — A+C 邏輯直接內聯於 `MS_SL2_split_model.forward()` 中
> - **沒有獨立的 `_get_adaptive_weight` 方法** — 融合計算直接內聯於 `forward()` 的 slice 迴圈內
> - **Pooling 來源是 `skip_connection`，不是 bottleneck `y`** — `c_skip = skip_connection.mean(dim=-1)`
> - **`fc_scale` 和 `fc_shift` 直接定義在 `__init__` 中**，不包裝成獨立類別
> - **`fc_scale` 和 `fc_shift` 都使用相同的輸入** `c_skip`（256 維，來自 `skip_connection.mean`）
> - **A+C 融合在每個 slice 的 TCN blocks 之後計算**（在 slice 迴圈內部）

> [!NOTE]
> 此計畫根據投影片規格更新：方向 A 完整流程為 **Pooling → FC → Sigmoid → Scaling**。

## 1. 核心概念

| 機制 | 解決的問題 | 類比 |
|------|-----------|------|
| **Phase Shift (C)** | 決定「誰配誰」— 哪個 weight 配哪個 slice | 旋轉組合鎖的起點 |
| **Input Scaling (A)** | 決定「各組多強」— 每個 weight 要保留或抑制多少 | 自動 EQ 調音量 |

### 1.1 現有問題回顧

目前程式碼（[model_SC_CHM_Fusion.py](file:///c:/Users/Wei/Downloads/denoiser/model_SC_CHM_Fusion.py) 第 592-606 行（原始版本））：

```python
# 每個 slice 都做一樣的事：
slic_into_weight = th.einsum('i,jkl->ijkl', [self.wList, skip_connection])
Slices_Output    = th.einsum('ijkl->jkl', [slic_into_weight])
# 結果 = (w₀+w₁+w₂+w₃) × skip_connection  ← 所有 slice 拿到同一個標量
```

> [!CAUTION]
> 4 個 weight 退化成一個標量 `Σwᵢ`，每個 slice 得到完全相同的加權。不同噪音環境下無法自適應。

### 1.2 A+C 修改後的公式

對每個 slice `s`，最終有效權重為：

```
w_eff(s) = Σ_{d=0}^{K-1}  δ_d · α_{(s+d)%K} · w_{(s+d)%K}
```

其中：
- `w_k` = 原有的 base weight（K=4 個，可學習）
- `δ` = Gumbel-Softmax 輸出 `(B, K)` — **決定 Δ 平移量**（Phase Shift）
- `α` = Input Scaling 輸出 `(B, K)` — **決定每個 weight 的保留比例**（Sigmoid，α ∈ (0, 1)）

---

## 2. 架構圖

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
         ┌───────────────────────────┐
         │   FOR each Slice s:      │
         │                          │
         │   ┌──────────────────┐   │
         │   │  TCN Blocks      │   │
         │   │  (R repeats ×    │   │
         │   │   X Conv1DBlock) │   │
         │   └────────┬─────────┘   │
         │            │             │
         │   skip_connection: (B, Sc, T)
         │            │             │
         │   ┌────────▼─────────┐   │
         │   │ c_skip = skip    │   │
         │   │   .mean(dim=-1)  │   │  ◄── Pooling 來源是 skip_connection
         │   │   → (B, 256)     │   │
         │   └───┬─────────┬────┘   │
         │       │         │        │
         │       ▼         ▼        │
         │   fc_scale   fc_shift    │  ◄── 兩者都接收 c_skip
         │   (256→4)   (256→4)     │
         │       │         │        │
         │   Sigmoid   Gumbel      │
         │       │     Softmax     │
         │       ▼         ▼        │
         │    α (B,4)   δ (B,4)    │
         │       │         │        │
         │       ▼         ▼        │
         │  ┌─────────────────────┐ │
         │  │ A+C Fusion (inline)│ │
         │  │ w_eff(s) = Σ_d     │ │
         │  │  δ_d·α[(s+d)%K]   │ │
         │  │  ·w[(s+d)%K]      │ │
         │  └─────────┬──────────┘ │
         │            │            │
         │   Slices_Output += w_eff│
         │     · skip_connection   │
         └────────────┬────────────┘
                      │ (B, Sc, T)
                      ▼
               PReLU → Mask → Decode
```

---

## 3. 實際修改的程式碼（共 2 處）

> [!NOTE]
> 實際實作中沒有獨立的 `AdaptiveController` 類別或 `_get_adaptive_weight` 方法。
> 所有 A+C 邏輯直接內聯於 `MS_SL2_split_model` 的 `__init__` 和 `forward` 中。

### 3.1 修改 `__init__` — 新增 `fc_scale`、`fc_shift`、`tau`

**位置：** [model_SC_CHM_Fusion.py 第 452-457 行](file:///c:/Users/Wei/Downloads/denoiser/model_SC_CHM_Fusion.py#L452-L457)，在 `self.wList = ...` 之後：

```python
# 方向 A: Input Scaling — FC(Sc→4), 額外參數: 256×4+4 = 1,028
self.fc_scale = nn.Linear(Sc, 4)

# 方向 C: Phase Shift — FC(N//2→4), 額外參數: 256×4+4 = 1,028
self.fc_shift = nn.Linear(N // 2, 4)   # N//2=256
self.tau = 1.0  # Gumbel-Softmax 溫度（在訓練中退火）
```

**說明：**
- `fc_scale` 和 `fc_shift` 都是 `nn.Linear(256, 4)`
- 沒有 `AdaptiveAvgPool1d` 層 — pooling 直接用 `.mean(dim=-1)` 在 `forward()` 中完成
- `tau` 是 Gumbel-Softmax 的溫度參數

**參數量：**
- `fc_scale`: 256×4 + 4 = 1,028
- `fc_shift`: 256×4 + 4 = 1,028
- **合計: 2,056 個參數 (< 3K ✓)**

### 3.2 修改 `forward` — 內聯 A+C 融合邏輯

**位置：** [model_SC_CHM_Fusion.py 第 562-603 行](file:///c:/Users/Wei/Downloads/denoiser/model_SC_CHM_Fusion.py#L562-L603)，slice 迴圈內部：

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

    # ── 方向 A: 用 skip_connection 做 pooling ──
    c_skip = skip_connection.mean(dim=-1)              # (B, Sc=256)
    alpha = th.sigmoid(self.fc_scale(c_skip))          # (B, 4)

    # ── 方向 C: 用 skip_connection 做 Gumbel-Softmax ──
    delta = F.gumbel_softmax(self.fc_shift(c_skip),
                              tau=self.tau, hard=not self.training, dim=-1)  # (B, 4)

    # ── A+C 融合: w_eff(s) = Σ_d δ_d · α[(s+d)%K] · w[(s+d)%K] ──
    indices = [(Slice + d) % K for d in range(K)]     # 循環索引
    shifted_w = self.wList[indices]                     # (K,)  phase-shifted base weights
    shifted_alpha = alpha[:, indices]                   # (B, K) phase-shifted scaling factors
    scaled_shifted = shifted_alpha * shifted_w          # (B, K) = αₖ · wₖ (shifted)
    w_eff = (delta * scaled_shifted).sum(dim=-1)        # (B,)
    weighted_skip = skip_connection * w_eff.unsqueeze(-1).unsqueeze(-1)  # (B, Sc, T)
    Slices_Output = Slices_Output + weighted_skip

    skip_connection = 0
    y = Slice_input
```

**關鍵設計：**
- **Pooling 來源是 `skip_connection`，不是 bottleneck `y`** — 因為 skip 已經累積了所有 TCN block 的資訊
- **`c_skip = skip_connection.mean(dim=-1)`** — 時間維度平均池化，產生 `(B, 256)` 向量
- **`fc_scale` 和 `fc_shift` 都接收 `c_skip`** — 同一個 256 維向量
- **A+C 融合在每個 slice 的 TCN blocks 之後計算** — 在 slice 迴圈內部，不是之前
- **推論時 `hard=not self.training`**（即 `hard=True`）— Gumbel-Softmax 用直通估計器

---

## 4. 張量形狀追蹤（具體數字）

以 `B=4, K=4, S=2 slices, Sc=256, T=1249` 為例：

```
步驟 1: Slice 0 — TCN blocks 完成後
────────────────────────────────────
skip_connection:  (4, 256, 1249)  ← 累積所有 TCN block 的 skip output

c_skip = skip_connection.mean(dim=-1)    # (4, 256) ← 時間維度平均池化
fc_scale(c_skip):  (4, 4)                ← Scaling logits
fc_shift(c_skip):  (4, 4)                ← Phase Shift logits
alpha = Sigmoid:   (4, 4)                = α，例如 [[0.8, 0.3, 0.9, 0.5], ...]
delta = GumbelSM:  (4, 4)                = δ，例如 [[0.01, 0.02, 0.95, 0.02], ...]

A+C 融合 (Slice=0):
  indices = [0, 1, 2, 3]
  shifted_w     = [w₀, w₁, w₂, w₃]               (4,)
  shifted_alpha = α[:, [0,1,2,3]]                 (4, 4)
  scaled_shifted = shifted_alpha * shifted_w       (4, 4)
  w_eff   = (δ · scaled_shifted).sum(dim=-1)       (4,)

  batch 0 的例子（δ ≈ [0,0,1,0], 即 Δ=2）：
    w_eff[0] ≈ 0 + 0 + 1·α₂·w₂ + 0 = α₂·w₂
    = 0.9 × w₂  ← Slice 0 用 w₂ 且保留 90%

步驟 2: Slice 1 — TCN blocks 完成後
────────────────────────────────────
skip_connection:  (4, 256, 1249)  ← 此 slice 重新累積
c_skip = skip_connection.mean(dim=-1)    # (4, 256)
（重新計算 alpha 和 delta — 每個 slice 獨立計算）

A+C 融合 (Slice=1):
  indices = [1, 2, 3, 0]
  shifted_w     = [w₁, w₂, w₃, w₀]               (4,)
  shifted_alpha = α[:, [1,2,3,0]]                 (4, 4)

  batch 0（δ ≈ [0,0,1,0], 即 Δ=2）：
    w_eff[0] ≈ 1·α₃·w₃ = 0.5 × w₃
    → Slice 1 用 w₃ 且保留 50%

結果:
  Slices_Output[batch 0] = 0.9·w₂·skip⁽⁰⁾ + 0.5·w₃·skip⁽¹⁾
```

---

## 5. 行為對比

```
┌────────────────────────────────────────────────────────┐
│                    現有模型                              │
│                                                        │
│  BUS 噪音:   Slice 0 → (w₀+w₁+w₂+w₃)·h⁽⁰⁾          │
│              Slice 1 → (w₀+w₁+w₂+w₃)·h⁽¹⁾          │
│  CAFE 噪音:  完全相同 ← 無法自適應!                     │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│               A+C 模型（Phase Shift + Scaling）         │
│                                                        │
│  BUS 噪音:  Δ=0, α=[0.9, 0.3, -, -]                  │
│    → Slice 0: 0.9·w₀·h⁽⁰⁾  (配 w₀，保留 90%)        │
│    → Slice 1: 0.3·w₁·h⁽¹⁾  (配 w₁，只保留 30%)      │
│                                                        │
│  CAFE 噪音: Δ=2, α=[-, -, 0.8, 0.5]                  │
│    → Slice 0: 0.8·w₂·h⁽⁰⁾  (配 w₂，保留 80%) ← 不同!│
│    → Slice 1: 0.5·w₃·h⁽¹⁾  (配 w₃，保留 50%) ← 不同!│
└────────────────────────────────────────────────────────┘
```

**關鍵差異：**
- Phase Shift 讓「配對方式」因輸入而異（Δ=0 vs Δ=2）
- Input Scaling 讓「保留比例」因輸入而異（α=0.9 vs α=0.8）
- 兩者互補：只有 C 的話強度固定；只有 A 的話配對固定

---

## 6. Gumbel-Softmax 溫度退火

| 訓練階段 | τ 值 | 行為 |
|---------|------|------|
| 初期 (epoch 1-10) | 1.0-2.0 | 軟選擇，探索所有 Δ |
| 中期 (epoch 10-30) | 0.5-1.0 | 逐漸聚焦 |
| 後期 (epoch 30-50) | 0.1-0.5 | 接近離散，確定 Δ |
| 推論 | hard=True | 精確 argmax |

**在 `trainer.py` 中加入（可選）：**

```python
# 在 run() 的 while 迴圈內，每個 epoch 開始前：
self.nnet.tau = max(0.1, 1.0 - (self.cur_epoch / num_epochs) * 0.9)
```

---

## 7. Sigmoid 的選擇理由（依投影片規格）

**投影片明確指定完整流程：Pooling → FC → Sigmoid → Scaling**

| 激活函數 | 範圍 | 特性 |
|---------|------|------|
| `Sigmoid(x)` | (0, 1) | **投影片指定** — α 語義為「保留比例」 |
| `Softplus(x)` | (0, +∞) | 允許放大，但可能數值不穩定 |
| `Sigmoid(x)·2` | (0, 2) | 有界，允許微幅放大，可作為 ablation 選項 |
| `exp(x)` | (0, +∞) | 容易爆炸，不推薦 |

> [!TIP]
> Sigmoid 的語義與「自動 EQ」類比一致：α ≈ 1.0 表示完全保留該 weight，α ≈ 0.0 表示幾乎抑制。
> 模型學會了「該保留哪些 weight、抑制哪些 weight」，而非「放大多少倍」。
> 如果實驗中發現需要放大能力，可嘗試 `Sigmoid(x) * 2` 作為 ablation。

---

## 8. Checkpoint 相容性

| 狀況 | 結果 |
|------|------|
| 載入舊 `best.pt.tar` + `strict=False` | ✅ 所有舊權重正常載入 |
| `fc_scale` 的新參數 | ✅ 隨機初始化（它是新增的） |
| `fc_shift` 的新參數 | ✅ 隨機初始化（它是新增的） |
| `wList` 的舊權重 | ✅ 正常繼承 |
| 舊的 `einsum` 邏輯 | ✅ 已被新邏輯替換，不影響載入 |

---

## 9. 修改檔案清單

| 檔案 | 修改 | 行數估計 |
|------|------|---------|
| `model_SC_CHM_Fusion.py` | `__init__` 新增 `fc_scale`、`fc_shift`、`tau` | +6 行 |
| `model_SC_CHM_Fusion.py` | `forward` 內聯 A+C 融合邏輯（替換 `einsum`） | ~20 行替換 ~15 行 |
| `trainer.py`（可選） | τ 退火 | +1 行 |
| `conf.py` | **不需修改** | 0 |

> [!NOTE]
> 沒有獨立的 `AdaptiveController` 類別或 `_get_adaptive_weight` 方法。
> 所有邏輯直接內聯，減少程式碼複雜度。

**總額外參數：2,056（< 3K ✓）**

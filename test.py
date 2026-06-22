#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test.py — 對測試集 (tt/) 進行語音分離推論並計算 SI-SNR

用法:
    python test.py \
        --checkpoint checkpoints/best.pt.tar \
        --output_dir ./test_output \
        --result_csv test_results.csv \
        --gpus 0

輸出:
    - 分離後的 WAV 檔 (output_dir/<key>_spk1.wav, _spk2.wav)
    - 每句 SI-SNR 與平均 SI-SNR 印在終端機
    - test_results.csv 彙整所有分數（可貼到 README）
"""

import os
import csv
import argparse
from itertools import permutations

import numpy as np
import torch as th

from audio import WaveReader, write_wav
from model_SC_CHM_Fusion import MS_SL2_split_model
from conf import nnet_conf, test_data

FS = 16000  # 採樣率


# ──────────────────────────────────────────────
# SI-SNR 計算
# ──────────────────────────────────────────────
def si_snr(x, s, eps=1e-8):
    """
    x: estimated signal  (numpy 1D)
    s: reference signal  (numpy 1D)
    return: SI-SNR (dB)
    """
    x = x - x.mean()
    s = s - s.mean()
    alpha = (x * s).sum() / (np.linalg.norm(s) ** 2 + eps)
    s_target = alpha * s
    noise = x - s_target
    return 20 * np.log10(np.linalg.norm(s_target) / (np.linalg.norm(noise) + eps) + eps)


def best_perm_sisnr(ests, refs):
    """
    ests: list of numpy arrays (num_spks,)
    refs: list of numpy arrays (num_spks,)
    return: best average SI-SNR (dB) over all permutations
    """
    num_spks = len(refs)
    best = -np.inf
    for perm in permutations(range(num_spks)):
        avg = np.mean([si_snr(ests[s], refs[t]) for s, t in enumerate(perm)])
        if avg > best:
            best = avg
    return best


# ──────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────
def run(args):
    # ── 載入模型 ──
    device = th.device(f"cuda:{args.gpus}" if th.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    nnet = MS_SL2_split_model(**nnet_conf)
    cpt = th.load(args.checkpoint, map_location="cpu")
    state = cpt.get("model_state_dict", cpt)
    nnet.load_state_dict(state, strict=False)
    nnet.to(device).eval()
    print(f"模型載入完成: {args.checkpoint}")

    # ── 讀取測試資料的 .scp（路徑來自 conf.test_data）──
    print(f"測試資料: {test_data['mix_scp']}")
    mix_reader  = WaveReader(test_data["mix_scp"],  sample_rate=FS)
    ref_readers = [WaveReader(ref_scp, sample_rate=FS) for ref_scp in test_data["ref_scp"]]

    # ── 建立輸出資料夾 ──
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    sisnr_list = []
    keys = mix_reader.index_keys

    print(f"\n共 {len(keys)} 句測試語音\n" + "=" * 50)

    with th.no_grad():
        for i, key in enumerate(keys):
            mix_np = mix_reader[key].astype(np.float32)   # (T,)
            refs_np = [r[key].astype(np.float32) for r in ref_readers]

            # 轉成 tensor (1, T)
            mix_t = th.tensor(mix_np, dtype=th.float32).unsqueeze(0).to(device)

            # 推論 — 模型輸出 list of (1, T) tensors
            ests = nnet(mix_t)

            # 轉成 numpy
            ests_np = [e.squeeze().cpu().numpy() for e in ests]

            # 對齊長度（取最短）
            L = min(mix_np.shape[0], *[r.shape[0] for r in refs_np], *[e.shape[0] for e in ests_np])
            ests_np = [e[:L] for e in ests_np]
            refs_np = [r[:L] for r in refs_np]

            # 計算最佳排列 SI-SNR
            score = best_perm_sisnr(ests_np, refs_np)
            sisnr_list.append(score)
            print(f"[{i+1:4d}/{len(keys)}] {key:30s}  SI-SNR = {score:+.2f} dB")

            # 儲存 WAV
            if args.output_dir:
                for s_idx, est in enumerate(ests_np):
                    out_path = os.path.join(args.output_dir, f"{key}_spk{s_idx+1}.wav")
                    peak = np.max(np.abs(est))
                    if peak > 0:
                        est = est / peak
                    write_wav(out_path, est, fs=FS)

    print("=" * 50)
    mean_sisnr = np.mean(sisnr_list)
    std_sisnr  = np.std(sisnr_list)
    print(f"平均 SI-SNR : {mean_sisnr:+.4f} dB")
    print(f"標準差      : {std_sisnr:.4f} dB")
    print(f"測試句數    : {len(sisnr_list)}")

    # ── 儲存 CSV 結果 ──
    if args.result_csv:
        with open(args.result_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Utterance", "SI-SNR (dB)"])
            for key, score in zip(keys, sisnr_list):
                writer.writerow([key, f"{score:.4f}"])
            writer.writerow(["**Average**", f"{mean_sisnr:.4f}"])
            writer.writerow(["**Std**",     f"{std_sisnr:.4f}"])
        print(f"\n結果已儲存至: {args.result_csv}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="對測試集進行語音分離推論並計算 SI-SNR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint",
                        type=str,
                        default="checkpoints/best.pt.tar",
                        help="訓練好的模型權重路徑 (.pt.tar)")
    parser.add_argument("--output_dir",
                        type=str,
                        default="",
                        help="分離後 WAV 儲存路徑，留空則不儲存")
    parser.add_argument("--result_csv",
                        type=str,
                        default="test_results.csv",
                        help="儲存每句 SI-SNR 的 CSV 路徑，留空則不儲存")
    parser.add_argument("--gpus",
                        type=int,
                        default=0,
                        help="使用的 GPU 編號（無 GPU 時自動改用 CPU）")
    args = parser.parse_args()
    run(args)

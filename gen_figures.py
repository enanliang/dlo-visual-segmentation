#!/usr/bin/env python3
"""Generate all 5 experimental figures from LTspice CSV data."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

def load(fname):
    df = pd.read_csv(os.path.join(DATA, fname))
    return df.iloc[:,0].values.astype(float), df.iloc[:,1].values.astype(float)

def ideal_sine(t, amp, freq):
    return amp * np.sin(2 * np.pi * freq * t)

# ── Fig 1: Sine comparison ────────────────────────────────────────────────
def fig1_sine():
    configs = [("bit4_sine.csv", "4-bit", "#e74c3c"),
               ("bit6_sine.csv", "6-bit", "#3498db"),
               ("bit8_sine.csv", "8-bit", "#2ecc71")]
    amp, freq = 1.0, 1000
    fig, ax = plt.subplots(figsize=(10, 4.5))
    t_ref, _ = load(configs[0][0])
    mask = t_ref <= 2e-3
    ax.plot(t_ref[mask]*1e6, ideal_sine(t_ref[mask], amp, freq), "k--", lw=0.8, alpha=0.5, label="Ideal")
    for fname, label, color in configs:
        t, v = load(fname)
        mse = np.mean((v - ideal_sine(t, amp, freq))**2)
        mask = t <= 2e-3
        ax.plot(t[mask]*1e6, v[mask], color=color, lw=0.7, alpha=0.85,
                label=f"{label} (MSE={mse:.2e})")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("V(OUT) (V)")
    ax.set_title("Sine Wave Quantization Comparison (1.0 V, 1000 Hz)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "sine_compare.png"))
    plt.close()
    print("  sine_compare.png")

# ── Fig 2: Step response ──────────────────────────────────────────────────
def fig2_step():
    configs = [("bit4_step.csv", "4-bit", "#e74c3c"),
               ("bit6_step.csv", "6-bit", "#3498db"),
               ("bit8_step.csv", "8-bit", "#2ecc71")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, (fname, label, color) in enumerate(configs):
        t, v = load(fname)
        t_us = t * 1e6
        ss_mask = (t >= 150e-6) & (t <= 200e-6)
        v_ss = np.mean(v[ss_mask])
        ax = axes[idx]
        ax.plot(t_us, v, color=color, lw=0.8)
        ax.axhline(v_ss, color="gray", ls="--", lw=0.5, alpha=0.7)
        ax.axhline(0.95*v_ss, color="orange", ls=":", lw=0.5, alpha=0.7)
        ax.axvspan(150, 200, color="yellow", alpha=0.12)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("V(OUT) (V)")
        ax.set_title(f"Step Response - {label}")
        ax.set_xlim(0, 250)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "step_compare.png"))
    plt.close()
    print("  step_compare.png")

# ── Fig 3: Amplitude sweep ────────────────────────────────────────────────
def fig3_amp():
    configs = [("amp_03.csv", 0.3), ("amp_06.csv", 0.6), ("amp_10.csv", 1.0)]
    freq = 1000
    amps, mses = [], []
    for fname, amp in configs:
        t, v = load(fname)
        ideal = ideal_sine(t, amp, freq)
        mse = np.mean((v - ideal)**2)
        amps.append(amp); mses.append(mse)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(amps, mses, "o-", color="#8e44ad", ms=8, lw=2)
    for a, m in zip(amps, mses):
        ax.annotate(f"{m:.2e}", (a, m), textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlabel("Input Amplitude (V)")
    ax.set_ylabel("MSE (V$^2$)")
    ax.set_title("MSE vs Input Amplitude (8-bit, 1000 Hz)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "amp_mse_curve.png"))
    plt.close()
    print("  amp_mse_curve.png")

# ── Fig 4: Sampling frequency ─────────────────────────────────────────────
def fig4_fs():
    configs = [("fs_500.csv", 500), ("fs_1000.csv", 1000), ("fs_2000.csv", 2000)]
    freq, amp = 1000, 1.0
    fs_list, mse_list, pwr_list = [], [], []
    for fname, fs in configs:
        t, v = load(fname)
        ideal = ideal_sine(t, amp, freq)
        mse = np.mean((v - ideal)**2)
        pwr = np.mean(np.abs(v))
        fs_list.append(fs); mse_list.append(mse); pwr_list.append(pwr)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    c1, c2 = "#c0392b", "#2980b9"
    ax1.plot(fs_list, mse_list, "s-", color=c1, ms=8, lw=2, label="MSE")
    ax1.set_xlabel("Sampling Frequency (Hz)")
    ax1.set_ylabel("MSE (V$^2$)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    for fs_val, m in zip(fs_list, mse_list):
        ax1.annotate(f"{m:.2e}", (fs_val, m), textcoords="offset points", xytext=(8, -12), fontsize=9, color=c1)
    ax2 = ax1.twinx()
    ax2.plot(fs_list, pwr_list, "D--", color=c2, ms=8, lw=2, label="Power Proxy")
    ax2.set_ylabel("Mean |V(OUT)| (V)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="upper center")
    plt.title("MSE and Power Proxy vs Sampling Frequency (8-bit, 1 V, 1000 Hz)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fs_curve.png"))
    plt.close()
    print("  fs_curve.png")

# ── Fig 5: Monte Carlo ────────────────────────────────────────────────────
def fig5_mc():
    df = pd.read_csv(os.path.join(DATA, "monte_carlo_results.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # (a) MSE hist
    ax = axes[0,0]
    ax.hist(df["mse"], bins=20, color="#3498db", edgecolor="white", alpha=0.85)
    m = df["mse"].mean()
    ax.axvline(m, color="red", ls="--", lw=1.5, label=f"Mean={m:.2e}")
    ax.set_xlabel("MSE (V$^2$)"); ax.set_ylabel("Count")
    ax.set_title("(a) MSE Distribution (100 runs)"); ax.legend(); ax.grid(True, alpha=0.3)
    # (b) Settling hist
    ax = axes[0,1]
    ax.hist(df["settling_us"], bins=20, color="#2ecc71", edgecolor="white", alpha=0.85)
    s = df["settling_us"].mean()
    ax.axvline(s, color="red", ls="--", lw=1.5, label=f"Mean={s:.2f} us")
    ax.set_xlabel("Settling Time (us)"); ax.set_ylabel("Count")
    ax.set_title("(b) Settling Time Distribution (100 runs)"); ax.legend(); ax.grid(True, alpha=0.3)
    # (c) Power hist
    ax = axes[1,0]
    ax.hist(df["power_mw"], bins=20, color="#e74c3c", edgecolor="white", alpha=0.85)
    p = df["power_mw"].mean()
    ax.axvline(p, color="blue", ls="--", lw=1.5, label=f"Mean={p:.3f} mW")
    ax.set_xlabel("Average Power (mW)"); ax.set_ylabel("Count")
    ax.set_title("(c) Power Distribution (100 runs)"); ax.legend(); ax.grid(True, alpha=0.3)
    # (d) Scatter
    ax = axes[1,1]
    sc = ax.scatter(df["settling_us"], df["mse"], c=df["power_mw"],
                    cmap="viridis", s=40, alpha=0.8, edgecolors="white", lw=0.3)
    plt.colorbar(sc, ax=ax, label="Power (mW)")
    ax.set_xlabel("Settling Time (us)"); ax.set_ylabel("MSE (V$^2$)")
    ax.set_title("(d) MSE vs Settling Time (color = Power)"); ax.grid(True, alpha=0.3)
    plt.suptitle("Monte Carlo: 8-bit CIM SAR-ADC under 5% DAC Capacitor Mismatch (100 runs)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "monte_carlo_distribution.png"))
    plt.close()
    print("  monte_carlo_distribution.png")

if __name__ == "__main__":
    print("Generating figures...")
    fig1_sine()
    fig2_step()
    fig3_amp()
    fig4_fs()
    fig5_mc()
    print("All figures saved to:", FIG)

#!/usr/bin/env python3
"""Generate simulated LTspice CSV data for 4 experiments."""
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
RNG = np.random.default_rng(42)

def save_csv(filename, time_arr, vout_arr):
    path = os.path.join(DATA_DIR, filename)
    np.savetxt(path, np.column_stack([time_arr, vout_arr]),
               delimiter=",", header="time,V(OUT)", comments="")
    print(f"  {filename} ({len(time_arr)} pts)")

def gen_step():
    dt = 1e-7
    t = np.arange(0, 250e-6, dt)
    v_final = 3.3
    configs = {
        "bit4_step.csv": {"bits": 4, "overshoot": 0.22, "settling_us": 45, "noise_std": 0.015},
        "bit6_step.csv": {"bits": 6, "overshoot": 0.10, "settling_us": 22, "noise_std": 0.006},
        "bit8_step.csv": {"bits": 8, "overshoot": 0.03, "settling_us": 8,  "noise_std": 0.0015},
    }
    for fname, cfg in configs.items():
        v = np.zeros_like(t)
        tau = cfg["settling_us"] * 1e-6 / 4.0
        for i in range(1, len(t)):
            v[i] = v[i-1] + (v_final - v[i-1]) / tau * dt
        wn = 2 * np.pi * 3e6
        zeta = 0.25 + cfg["overshoot"]
        ring = cfg["overshoot"] * v_final * np.exp(-zeta * wn * t) * np.cos(wn * np.sqrt(1 - zeta**2) * t)
        v += ring
        levels = 2 ** cfg["bits"]
        step_size = v_final / levels
        v = np.round(v / step_size) * step_size
        v += RNG.normal(0, cfg["noise_std"], len(t))
        save_csv(fname, t, v)

def gen_sine():
    dt = 1e-7
    t = np.arange(0, 2e-3, dt)
    freq = 1000
    amp = 1.0
    configs = {
        "bit4_sine.csv": {"bits": 4, "noise_std": 0.012},
        "bit6_sine.csv": {"bits": 6, "noise_std": 0.004},
        "bit8_sine.csv": {"bits": 8, "noise_std": 0.001},
    }
    for fname, cfg in configs.items():
        ideal = amp * np.sin(2 * np.pi * freq * t)
        delay = 0.5e-6 * (8 - cfg["bits"] + 4)
        delayed = np.interp(t - delay, t, ideal)
        levels = 2 ** cfg["bits"]
        step_size = (2 * amp) / levels
        quantized = np.round(delayed / step_size) * step_size
        v = quantized + RNG.normal(0, cfg["noise_std"], len(t))
        save_csv(fname, t, v)

def gen_amp():
    dt = 1e-7
    t = np.arange(0, 2e-3, dt)
    freq = 1000
    bits = 8
    amps = {"amp_03.csv": 0.3, "amp_06.csv": 0.6, "amp_10.csv": 1.0}
    for fname, amp in amps.items():
        ideal = amp * np.sin(2 * np.pi * freq * t)
        levels = 2 ** bits
        step_size = (2 * amp) / levels
        quantized = np.round(ideal / step_size) * step_size
        v = quantized + RNG.normal(0, 0.001, len(t))
        save_csv(fname, t, v)

def gen_fs():
    freq_sig = 1000
    amp = 1.0
    bits = 8
    duration = 20e-3
    configs = {"fs_500.csv": 500, "fs_1000.csv": 1000, "fs_2000.csv": 2000}
    for fname, fs in configs.items():
        n_pts = int(duration * fs) + 1
        t = np.linspace(0, duration, n_pts, endpoint=False)
        ideal = amp * np.sin(2 * np.pi * freq_sig * t)
        levels = 2 ** bits
        step_size = (2 * amp) / levels
        quantized = np.round(ideal / step_size) * step_size
        v = quantized + RNG.normal(0, 0.001, n_pts)
        save_csv(fname, t, v)

if __name__ == "__main__":
    print("Generating simulated LTspice CSV data...")
    gen_step(); gen_sine(); gen_amp(); gen_fs()
    print("Done.")

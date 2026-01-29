import os
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- Ensure results folder exists ---
os.makedirs("results", exist_ok=True)

print("\n🐶 Running Audio Prototype (YAMNet-inspired energy detector)...\n")

# --- Load audio ---
AUDIO_FILE = "dog_cough.wav"

if not os.path.exists(AUDIO_FILE):
    print(f"❌ ERROR: File '{AUDIO_FILE}' not found.")
    exit()

y, sr = librosa.load(AUDIO_FILE, sr=None)
duration = len(y) / sr

print(f"Loaded audio: {AUDIO_FILE}")
print(f"Sample rate: {sr} Hz")
print(f"Length: {duration:.2f} seconds\n")

# --- Step 1: Compute energy ---
frame_length = 2048
hop_length = 512
energy = np.array([
    sum(abs(y[i:i+frame_length]**2))
    for i in range(0, len(y), hop_length)
])

# --- Step 2: Detect bursts (pseudo-coughs) ---
threshold = np.mean(energy) + 1.5*np.std(energy)
cough_frames = np.where(energy > threshold)[0]

# Convert to timestamps
cough_timestamps = (cough_frames * hop_length) / sr
cough_timestamps = np.round(cough_timestamps, 2).tolist()

print(f"Detected high-energy bursts (cough-like events): {len(cough_timestamps)}")
print("Approx cough timestamps (s):")
print(cough_timestamps, "\n")

# --- Step 3: Simple risk scoring ---
risk_score = min(1.0, len(cough_timestamps) * 0.07)
label = "High" if risk_score > 0.6 else "Moderate" if risk_score > 0.3 else "Low"

print(f"Estimated kennel cough risk: {label} (score ≈ {risk_score:.2f})\n")

# --- Step 4: Save JSON with timestamp (FIXED VERSION) ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = os.path.join("results", f"audio_analysis_{timestamp}.json")

analysis = {
    "risk_score": risk_score,          # 👈 generic key used by fusion_test.py
    "audio_risk_score": risk_score,    # 👈 optional, nice for clarity
    "risk_label": label,
    "sample_rate": sr,
    "num_cough_events": len(cough_timestamps),
    "cough_timestamps": cough_timestamps
}


with open(json_path, "w") as f:
    json.dump(analysis, f, indent=2)

print(f"💾 Saved analysis JSON to: {json_path}\n")

# --- Step 5: Save waveform plot ---
plt.figure(figsize=(10, 3))
plt.plot(y)
plt.title("Waveform")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
wave_path = os.path.join("results", "waveform.png")
plt.savefig(wave_path)
plt.close()
print(f"💾 Saved waveform plot to: {wave_path}")

# --- Step 6: Save energy plot ---
plt.figure(figsize=(10, 3))
plt.plot(energy)
plt.title("Energy Curve")
plt.xlabel("Frame")
plt.ylabel("Energy")
energy_path = os.path.join("results", "energy.png")
plt.savefig(energy_path)
plt.close()
print(f"💾 Saved energy plot to: {energy_path}\n")



def run_audio(audio_path: str) -> dict:
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=None)

    frame_length = 2048
    hop_length = 1024

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Smooth RMS
    win = 5
    kernel = np.ones(win) / win
    rms_smooth = np.convolve(rms, kernel, mode="same")

    # Robust stats
    med = np.median(rms_smooth)
    mad = np.median(np.abs(rms_smooth - med)) + 1e-9

    # Peak spacing
    min_event_distance_s = 0.5
    min_distance_frames = int(min_event_distance_s * sr / hop_length)

    def detect_peaks(thr: float):
        peaks = []
        for i in range(1, len(rms_smooth) - 1):
            is_peak = (
                (rms_smooth[i] > thr) and
                (rms_smooth[i] > rms_smooth[i - 1]) and
                (rms_smooth[i] > rms_smooth[i + 1])
            )
            if is_peak:
                if len(peaks) == 0 or (i - peaks[-1]) >= min_distance_frames:
                    peaks.append(i)
        return peaks

    # --- thresholds: main + fallbacks ---
    thresholds = [
        med + 10.0 * mad,  # k
        med + 8.0 * mad,   # k2
        med + 6.0 * mad    # k3 (new, only if needed)
    ]

    peak_indices = []
    for thr in thresholds:
        peak_indices = detect_peaks(thr)
        if len(peak_indices) > 0:
            break

    # Convert peaks to timestamps
    cough_timestamps = (np.array(peak_indices) * hop_length) / sr
    cough_timestamps = np.round(cough_timestamps, 2).tolist()
    num_events = len(cough_timestamps)

    # --- Improved scoring ---
    duration_s = len(y) / sr
    events_per_10s = num_events / max(1e-9, (duration_s / 10.0))

    # rate + count components
    rate_score = min(1.0, events_per_10s / 6.0)
    count_score = min(1.0, num_events / 25.0)

    raw_score = min(1.0, 0.5 * rate_score + 0.5 * count_score)

    # ✅ NEW: reduce barking false positives (small number of peaks shouldn't spike)
    if num_events < 4:
        raw_score *= 0.4   # damp small-peak cases hard

    risk_score = raw_score

    return {
        "cough_score": round(risk_score, 3),
        "events": num_events,
        "timestamps": cough_timestamps
    }

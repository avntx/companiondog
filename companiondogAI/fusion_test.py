import os
import glob
import json
from datetime import datetime

RESULTS_DIR = "results"


def load_latest_json(pattern: str):
    """Return (latest_file_path, loaded_json) or (None, None) if not found."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not files:
        return None, None
    latest = files[-1]
    try:
        with open(latest, "r") as f:
            data = json.load(f)
        return latest, data
    except Exception as e:
        print(f"⚠️  Error reading {latest}: {e}")
        return latest, None


def main():
    print("\n🐾 Fusion Prototype — Combining Audio + Text Risk Scores\n")

    # 🔊 AUDIO: accept both audio_analysis.json AND audio_analysis_*.json
    audio_file, audio_data = load_latest_json("audio_analysis*.json")
    if audio_file and audio_data:
        audio_score = audio_data.get("risk_score", None)
        print(f"🎧 Latest audio file: {audio_file}")
        print(f"   Audio risk score: {audio_score}")
    else:
        audio_score = None
        print("⚠️  No usable audio_analysis JSON found in results/. "
              "Run audio_test.py first or check the file content.")

    # 📝 TEXT: use latest text_analysis_*.json
    text_file, text_data = load_latest_json("text_analysis_*.json")
    if text_file and text_data:
        text_score = text_data.get("text_risk_score", text_data.get("risk_score", None))
        print(f"📝 Latest text file:  {text_file}")
        print(f"   Text risk score:  {text_score}")
    else:
        text_score = None
        print("⚠️  No usable text_analysis JSON found in results/. "
              "Run text_prototype.py first.")

    # 🧮 FUSION LOGIC
    scores = [s for s in [audio_score, text_score] if isinstance(s, (int, float))]
    if scores:
        fused_score = sum(scores) / len(scores)
    else:
        fused_score = None

    # Simple label mapping
    def label(score):
        if score is None:
            return "Unknown"
        if score >= 0.75:
            return "High"
        if score >= 0.4:
            return "Medium"
        return "Low"

    fused_label = label(fused_score)

    print("\n🧮 Fused Risk:")
    print(f"   Score: {fused_score} ({fused_label})\n")

    print("📄 Summary:")
    print("Based on the available inputs:")
    print(f"- Audio-based risk: {audio_score if audio_score is not None else 'N/A'}")
    print(f"- Text-based risk:  {text_score if text_score is not None else 'N/A'}\n")
    print(f"Overall kennel cough likelihood: {fused_score} ({fused_label}).\n")

    # 💾 SAVE SUMMARY JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(RESULTS_DIR, f"fusion_summary_{timestamp}.json")

    summary = {
        "audio_file": audio_file,
        "audio_risk_score": audio_score,
        "text_file": text_file,
        "text_risk_score": text_score,
        "fused_risk_score": fused_score,
        "fused_risk_label": fused_label,
        "timestamp": timestamp,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Saved fusion summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()


def run_fusion(audio_out: dict, vision_out: dict, text_out: dict) -> dict:
    """
    Returns a dict like:
    { "risk_score": 0.0-1.0, "risk_level": "Low/Medium/High", "explanation": [...] }
    """
    a = float(audio_out.get("cough_score", 0.0))
    v = vision_out.get("dog_conf", 0.0) if vision_out.get("dog_detected") else 0.0
    t = float(text_out.get("severity_score", 0.0))

    # Simple weighted fusion (you can tune later)
    risk = (0.6 * a) + (0.2 * v) + (0.2 * t)

    if risk > 0.5:
        level = "High"
    elif risk >= 0.2:
        level = "Medium"
    else:
        level = "Low"

    explanation = []
    if a >= 0.6:
        explanation.append("Audio: cough-like events detected.")
    if vision_out.get("dog_detected", False):
        explanation.append("Vision: dog detected in image.")
    if t > 0:
        explanation.append(f"Text: symptoms mentioned ({', '.join(text_out.get('keywords', []))}).")

    if not explanation:
        explanation.append("No strong signals found from inputs.")

    return {
        "risk_score": round(risk, 3),
        "risk_level": level,
        "explanation": explanation
    }

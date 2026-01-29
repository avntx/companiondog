import os
import uuid
import streamlit as st

from audio_test import run_audio
from vision_test import run_vision
from text_prototype import run_text
from fusion_test import run_fusion

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="CompanionDogAI",
    layout="centered"
)

st.title("🐶 CompanionDogAI")
st.write("Early kennel cough risk assessment using audio, vision, and owner input.")
st.caption(
    "⚠️ This system provides a non-diagnostic risk indication and does not replace professional veterinary advice."
)
st.markdown("---")

# ---------------------------
# Input section
# ---------------------------
st.header("Upload Inputs")

audio_file = st.file_uploader(
    "Upload dog audio (WAV format)",
    type=["wav"]
)

image_file = st.file_uploader(
    "Upload dog image (optional)",
    type=["jpg", "jpeg", "png"]
)

owner_note = st.text_area(
    "Describe your dog's symptoms or recent behaviour (optional)",
    placeholder="e.g. coughing after walks, low energy, daycare exposure"
)

st.markdown("---")

# ---------------------------
# Previews
# ---------------------------
if image_file is not None:
    st.image(image_file, caption="Uploaded image", use_column_width=True)

if audio_file is not None:
    st.success(f"Audio uploaded: {audio_file.name}")

if owner_note.strip() == "":
    st.caption("Tip: Adding a short description improves interpretability.")

can_run = audio_file is not None

# ---------------------------
# Run analysis
# ---------------------------
if st.button("Assess Risk", disabled=not can_run):
    st.write("Running analysis...")

    # Save uploads
    os.makedirs("tmp_uploads", exist_ok=True)
    run_id = str(uuid.uuid4())[:8]

    audio_path = os.path.join("tmp_uploads", f"{run_id}_audio.wav")
    with open(audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    image_path = None
    if image_file is not None:
        ext = os.path.splitext(image_file.name)[-1].lower()
        image_path = os.path.join("tmp_uploads", f"{run_id}_image{ext}")
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())

    # ---------------------------
    # Run individual modules
    # ---------------------------
    audio_out = run_audio(audio_path)

    if image_path is not None:
        vision_out = run_vision(image_path)
    else:
        vision_out = {
            "dog_detected": False,
            "dog_conf": 0.0,
            "notes": "No image provided"
        }

    text_out = run_text(owner_note)

    fusion_out = run_fusion(audio_out, vision_out, text_out)

    # ---------------------------
    # Results
    # ---------------------------
    st.success("Risk Assessment Complete")

    st.metric("Estimated Risk Level", fusion_out["risk_level"])
    st.metric("Risk Score (0–1)", round(fusion_out["risk_score"], 3))

    # ---------------------------
    # Explanation
    # ---------------------------
    st.subheader("Explanation")
    for line in fusion_out["explanation"]:
        st.write(f"- {line}")

    # =====================================================
    # ⭐ OPTION B — Risk Breakdown Visualisation
    # =====================================================
    st.subheader("Risk Contribution Breakdown")

    audio_score = float(audio_out.get("cough_score", 0.0))
    vision_score = float(vision_out.get("dog_conf", 0.0))
    text_score = float(text_out.get("severity_score", 0.0))

    total = audio_score + vision_score + text_score + 1e-9

    audio_pct = audio_score / total
    vision_pct = vision_score / total
    text_pct = text_score / total

    st.write("Relative contribution of each modality:")

    st.progress(audio_pct)
    st.write(f"Audio: {audio_pct * 100:.1f}%")

    st.progress(vision_pct)
    st.write(f"Vision: {vision_pct * 100:.1f}%")

    st.progress(text_pct)
    st.write(f"Text: {text_pct * 100:.1f}%")

    # =====================================================
    # ⭐ OPTION C — Session Summary
    # =====================================================
    st.subheader("Session Summary")

    risk_level = fusion_out["risk_level"]

    if risk_level == "Low":
        summary = (
            "Based on the current inputs, your dog shows a **low respiratory risk**. "
            "No strong cough patterns were detected. Continue monitoring your dog and "
            "consult a veterinarian if symptoms persist or worsen."
        )
    elif risk_level == "Medium":
        summary = (
            "Based on the current inputs, your dog shows a **moderate respiratory risk**. "
            "Some cough-like patterns were detected. Monitoring is recommended, and "
            "veterinary advice should be considered if symptoms continue."
        )
    else:
        summary = (
            "Based on the current inputs, your dog shows a **high respiratory risk**. "
            "Strong and sustained cough-like patterns were detected. Veterinary "
            "consultation is strongly advised."
        )

    st.info(summary)

    # ---------------------------
    # Debug / Evidence Outputs
    # ---------------------------
    with st.expander("Raw outputs (for debugging / report evidence)"):
        st.write("Audio output:", audio_out)
        st.write("Vision output:", vision_out)
        st.write("Text output:", text_out)

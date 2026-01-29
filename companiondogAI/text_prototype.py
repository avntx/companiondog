import os
import json
from datetime import datetime

RESULTS_DIR = "results"


def analyze_owner_note(note: str) -> dict:
    """
    Very simple text analysis:
    - lowercases text
    - looks for symptom and context keywords
    - produces a crude text risk score between 0 and 1
    """
    text = note.lower()

    symptom_keywords = [
        "cough", "hacking", "gagging", "wheezing",
        "honking", "choking", "phlegm", "mucus"
    ]
    context_keywords = [
        "daycare", "shelter", "boarding", "grooming",
        "park", "dog park", "kennel"
    ]
    severity_keywords = [
        "lethargic", "tired", "not eating", "no appetite",
        "weak", "breathing fast", "difficulty breathing"
    ]

    found_symptoms = [w for w in symptom_keywords if w in text]
    found_context = [w for w in context_keywords if w in text]
    found_severity = [w for w in severity_keywords if w in text]

    # Very simple scoring: more hits → higher risk
    score = 0.0
    if found_symptoms:
        score += 0.4
    if found_context:
        score += 0.3
    if found_severity:
        score += 0.3
    score = min(score, 1.0)

    risk_label = (
        "High"
        if score >= 0.7 else
        "Medium"
        if score >= 0.4 else
        "Low"
    )

    return {
        "note": note,
        "found_symptoms": found_symptoms,
        "found_context": found_context,
        "found_severity": found_severity,
        "text_risk_score": round(score, 2),
        "text_risk_label": risk_label,
    }


def save_result(result: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"text_analysis_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_path


def main():
    print("🐾 Text Prototype — Owner Note Analysis")
    print("Type a short description of your dog's condition.")
    print("Example: 'My dog has a hacking cough after daycare and is very tired.'\n")

    note = input("Owner note: ").strip()
    if not note:
        print("No note entered. Exiting.")
        return

    result = analyze_owner_note(note)

    print("\n🔍 Extracted Information:")
    print(f"  Symptoms found: {result['found_symptoms']}")
    print(f"  Context found:  {result['found_context']}")
    print(f"  Severity words: {result['found_severity']}")
    print(f"\n🧮 Text-based risk score: {result['text_risk_score']} ({result['text_risk_label']})")

    out_path = save_result(result)
    print(f"\n💾 Saved text analysis JSON to: {out_path}")


if __name__ == "__main__":
    main()




def run_text(owner_note: str) -> dict:
    """
    Rule-based NLP with:
    - keyword/phrase detection
    - negation handling (e.g., 'no cough')
    - intensity modifiers (e.g., 'frequent', 'multiple times', 'at night', 'dry hacking')
    Returns:
      {
        "keywords": [...],
        "modifiers_detected": [...],
        "severity_score": 0..1,
        "notes": "..."
      }
    """
    import re

    text = (owner_note or "").strip().lower()
    if not text:
        return {
            "keywords": [],
            "modifiers_detected": [],
            "severity_score": 0.0,
            "notes": "No owner note provided"
        }

    # --- helper: detect negated mentions ---
    def is_negated(term: str) -> bool:
        patterns = [
            rf"\bno\s+{term}\b",
            rf"\bnot\s+{term}\b",
            rf"\bwithout\s+{term}\b",
            rf"\bdoesn['’]?t\s+{term}\b",
            rf"\bdidn['’]?t\s+{term}\b",
        ]
        return any(re.search(p, text) for p in patterns)

    # --- base concepts and patterns ---
    concepts = {
        "cough": [r"cough", r"coughing", r"hacking", r"hack"],
        "gagging": [r"gag", r"gagging", r"retch", r"retching"],
        "sneeze": [r"sneeze", r"sneezing"],
        "lethargy": [r"lethargy", r"lethargic", r"low energy", r"tired", r"fatigue", r"weak"],
        "daycare": [r"daycare", r"kennel", r"boarding", r"grooming"],
        "appetite_loss": [r"not eating", r"loss of appetite", r"no appetite", r"poor appetite"],
    }

    detected = []

    # detect concepts (with negation handling for symptom concepts)
    for concept, pats in concepts.items():
        found = False
        for pat in pats:
            if re.search(rf"\b{pat}\b", text):
                if concept in {"cough", "gagging", "sneeze", "lethargy", "appetite_loss"}:
                    token = pat.split()[0]
                    if is_negated(token):
                        found = False
                        continue
                found = True
                break
        if found:
            detected.append(concept)

    # Deduplicate while preserving order
    keywords = list(dict.fromkeys(detected))

    # --- intensity modifiers ---
    modifiers = []
    intensity = 0.0

    # frequency / repetition
    if re.search(r"\bfrequent\b|\boften\b|\brepeated\b|\bmany\b|\bseveral\b", text):
        modifiers.append("frequent")
        intensity += 0.15

    if re.search(r"\bmultiple times\b|\bevery day\b|\bdaily\b", text):
        modifiers.append("daily/multiple")
        intensity += 0.15

    # night / persistent
    if re.search(r"\bat night\b|\bnight\b|\bkeeping us up\b", text):
        modifiers.append("night")
        intensity += 0.10

    # cough quality descriptors
    if re.search(r"\bdry\b|\bhacking\b|\bchoking\b", text):
        modifiers.append("dry/hacking")
        intensity += 0.10

    # --- weights (transparent scoring) ---
    weights = {
        "cough": 0.25,
        "gagging": 0.20,
        "sneeze": 0.10,
        "lethargy": 0.20,
        "daycare": 0.10,
        "appetite_loss": 0.25
    }

    severity = sum(weights.get(k, 0.10) for k in keywords) + intensity
    severity_score = min(1.0, round(severity, 3))

    return {
        "keywords": keywords,
        "modifiers_detected": modifiers,
        "severity_score": severity_score,
        "notes": "Rule-based NLP with negation + intensity modifiers (interpretable)"
    }

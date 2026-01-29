#!/bin/bash

echo "🐶 Running CompanionDogAI Full Prototype Pipeline"
echo "------------------------------------------------"

echo "🔊 Running Audio Prototype..."
python audio_test.py

echo ""
echo "📝 Running Text Prototype..."
python text_prototype.py

echo ""
echo "📷 Running Vision Prototype..."
python vision_test.py

echo ""
echo "🧠 Running Fusion Prototype..."
python fusion_test.py

echo ""
echo "✅ All modules executed successfully!"

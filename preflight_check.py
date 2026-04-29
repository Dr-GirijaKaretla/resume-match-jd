import os
import importlib

print("\n=== PREFLIGHT CHECK ===\n")

# 1. API key
if os.getenv("GEMINI_API_KEY"):
    print("✔ Gemini API key found")
else:
    print("✖ Gemini API key missing")

# 2. Required packages
required = [
    "streamlit",
    "google.generativeai",
    "PyPDF2",
    "docx"
]

for pkg in required:
    try:
        importlib.import_module(pkg)
        print(f"✔ {pkg} installed")
    except ImportError:
        print(f"✖ {pkg} NOT installed")

# 3. Conflicts
conflicts = [
    "transformers",
    "huggingface_hub",
    "accelerate",
    "peft",
    "diffusers",
    "sentence_transformers"
]

for pkg in conflicts:
    try:
        importlib.import_module(pkg)
        print(f"⚠ WARNING: Conflicting package detected → {pkg}")
    except ImportError:
        pass

print("\nPreflight check complete.\n")

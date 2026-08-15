#!/usr/bin/env bash
# Convert sam8000/whisper-large-v3-turbo-maltese-malta to CTranslate2 so
# faster-whisper can load it.
#
# Benchmarked and NOT adopted — see the README. It is a quarter the size of the
# large Maltese fine-tune but only ~9% faster, because Whisper pads every input to
# 30s and turbo only trims the decoder. Kept because it is the obvious thing to try
# and this saves the next person the hour.
set -euo pipefail
cd "$(dirname "$0")/.."

./.venv/bin/ct2-transformers-converter \
  --model sam8000/whisper-large-v3-turbo-maltese-malta \
  --output_dir models/whisper-turbo-maltese-ct2 \
  --copy_files preprocessor_config.json \
  --quantization int8 --force

# The source repo ships vocab.json + merges.txt but no tokenizer.json, which
# faster-whisper wants; build one.
./.venv/bin/python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(
    "sam8000/whisper-large-v3-turbo-maltese-malta"
).save_pretrained("models/whisper-turbo-maltese-ct2")
print("tokenizer written")
PY

echo "→ compare it:"
echo "   python scripts/compare_stt.py --models models/whisper-turbo-maltese-ct2,carlosdanielhernandezmena/whisper-large-maltese-8k-steps-64h-ct2"

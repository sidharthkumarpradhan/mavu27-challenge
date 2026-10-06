PY ?= python3

.PHONY: install test smoke board

install:
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev,remote]" "transformers==5.19.0" "peft==0.21.2" "av==19.0.1" accelerate pillow huggingface_hub

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m reva.cli smoke --out work/smoke

board:
	$(PY) -m reva.cli board

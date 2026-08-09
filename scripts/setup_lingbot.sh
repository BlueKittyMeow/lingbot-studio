#!/bin/bash
exec > ~/lingbot_setup.log 2>&1
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lingbot-map
echo "=== STEP: torch repair (force-reinstall)"
pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('TORCH_OK', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" || { echo "STEP_FAIL torch-import"; echo "SETUP_COMPLETE"; exit 1; }
cd ~/lingbot-map
echo "=== STEP: pip install -e ."
pip install -e . && echo "STEP_OK core" || echo "STEP_FAIL core"
echo "=== STEP: vis extras"
pip install -e ".[vis]" && echo "STEP_OK vis" || echo "STEP_FAIL vis"
echo "=== STEP: flashinfer"
pip install --index-url https://pypi.org/simple flashinfer-python && echo "STEP_OK flashinfer" || echo "STEP_FAIL flashinfer"
echo "=== STEP: checkpoint download"
pip install -q "huggingface_hub[cli]"
mkdir -p ~/models
hf download robbyant/lingbot-map --local-dir ~/models/lingbot-map && echo "STEP_OK checkpoint" || echo "STEP_FAIL checkpoint"
ls -la ~/models/lingbot-map
echo "SETUP_COMPLETE"

python -m uvicorn main:app --reload

pip install torch
pip install -r requirements.txt


window일 경우
pip uninstall torch torchvision torchaudio
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt


mac일 경우
pip install torch --index-url https://download.pytorch.org/whl/cpu
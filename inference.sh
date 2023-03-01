# conda activate diff
cd /mnt/proj73/taohu/Program/diffusers
pip install -e ".[torch]"
cd -

python inference.py
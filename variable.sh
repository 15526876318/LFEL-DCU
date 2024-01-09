#conda create --name torch_t1 python==3.6.13
conda activate torch_t1
pip install -r requirements.txt
pip install torch-1.8.0+rocm4.0.1-cp36-cp36m-linux_x86_64.whl

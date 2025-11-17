rlaunch --gpu=1 --memory=12800 --cpu=64 \
--charged-group=ai4good1_gpu --private-machine=yes \
--mount=gpfs://gpfs1/yuyi:/mnt/shared-storage-user/yuyi \
-- bash -c "cd ~/code/LLMOPT && PYTHONPATH=~/code/LLMOPT:~/code/LLMOPT/prompts python3 inference/inference_q2f2c.py && sleep inf"

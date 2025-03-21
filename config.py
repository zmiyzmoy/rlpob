# config.py
import torch
import os
import logging
import torch.multiprocessing as mp  # Import multiprocessing


# ========== КОНФИГУРАЦИЯ ==========
model_path = '/home/gunelmikayilova91/rlcard/pai.pt'
best_model_path = '/home/gunelmikayilova91/rlcard/pai_best.pt'
log_dir = '/home/gunelmikayilova91/rlcard/logs/'
os.makedirs(log_dir, exist_ok=True)  # Make sure log_dir exists
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'training.log')),
        logging.StreamHandler()
    ]
)


num_episodes = 10
batch_size = 512
gamma = 0.96
epsilon_start = 1.0
epsilon_end = 0.01
target_update_freq = 1000
buffer_capacity = 500_000
num_workers = 1
steps_per_worker = 1000
selfplay_update_freq = 5000
log_freq = 25
test_interval = 2000
early_stop_patience = 1000
early_stop_threshold = 0.001
learning_rate = 1e-4
noise_scale = 0.1
num_tables = 1
num_players = 6
code_version = "v1.9_test"
grad_clip_value = 5.0

device = torch.device("cpu")

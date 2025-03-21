# main.py

import torch
import torch.optim as optim
from models.model import DuelingPokerNN
from utils.environment_utils import init_env, get_state_size, run_game
from replay_buffer import PrioritizedExperienceBuffer
from utils.training_utils import train
import logging
from config import (num_episodes, batch_size, target_update_freq, log_freq,
                    learning_rate, model_path, best_model_path, test_interval, code_version, device)  # Import from config
from torch.utils.tensorboard import SummaryWriter
from utils.action_utils import ActionAbstraction
import os  # Import os

if __name__ == "__main__":
    env = init_env()
    state_size = get_state_size(env)
    action_abstraction = ActionAbstraction()
    num_actions = action_abstraction.num_actions

    # Initialize models, optimizers, and replay buffer
    models = [DuelingPokerNN(state_size, num_actions).to(device) for _ in range(env.num_players * 2)]  # Online and target networks
    optimizers = [optim.AdamW(models[i].parameters(), lr=learning_rate) for i in range(0, len(models), 2)]
    replay_buffer = PrioritizedExperienceBuffer()

    # Load existing model if it exists
    if os.path.exists(model_path):
        try:
            models[0].load_state_dict(torch.load(model_path))
            logging.info("Loaded model from checkpoint.")
            for model in models[1:]:  # Update all target networks
                model.load_state_dict(models[0].state_dict())
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            # Initialize models here if load fails
            for model in models:
                model = DuelingPokerNN(state_size, num_actions).to(device)

    else:  # Initialize models if no checkpoint exists
        for model in models:
                model = DuelingPokerNN(state_size, num_actions).to(device)


    # Initialize TensorBoard writer
    log_dir_tensorboard = f"/home/gunelmikayilova91/rlcard/logs/tensorboard/{code_version}"
    writer = SummaryWriter(log_dir=log_dir_tensorboard)


    # Train the agent
    train(env, models, optimizers, replay_buffer, writer, num_episodes, batch_size, target_update_freq, log_freq, test_interval)

    # Close the environment and TensorBoard writer
    env.close()
    writer.close()
    logging.info("Training completed.")

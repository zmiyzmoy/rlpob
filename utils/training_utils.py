# utils/training_utils.py

import logging
import torch
import torch.nn.functional as F
from config import gamma, target_update_freq, device, grad_clip_value, best_model_path
from models.model import DuelingPokerNN
from .action_utils import get_action_probs, fast_normalize
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .environment_utils import run_game, init_env
from replay_buffer import PrioritizedExperienceBuffer
from copy import deepcopy  # Import deepcopy



def compute_td_errors(batch, dqn_model, target_dqn_model):
    states, actions, rewards, next_states, dones, legal_actions_masks, player_ids = zip(*batch)

    states_tensor = torch.FloatTensor(np.array(states)).to(device)
    actions_tensor = torch.LongTensor(np.array(actions)).to(device)
    rewards_tensor = torch.FloatTensor(np.array(rewards)).to(device)
    
    non_final_mask = torch.tensor([not done for done in dones], dtype=torch.bool).to(device)
    non_final_next_states = [ns for ns, done in zip(next_states, dones) if not done]
    
    if not non_final_next_states:  # Handle empty list
        return torch.zeros_like(rewards_tensor).to(device)
        
    non_final_next_states_tensor = torch.FloatTensor(np.array(non_final_next_states)).to(device)

    # Compute Q-values
    q_values = dqn_model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    
    # Next Q-values (using target network and double DQN logic)
    next_q_values_target = torch.zeros(len(batch)).to(device)
    
    # Get next actions from the online network for Double DQN
    next_state_q_online = dqn_model(non_final_next_states_tensor)

    # Get legal actions for each of the next states
    next_legal_actions_masks = [mask for mask, done in zip(legal_actions_masks, dones) if not done]

    # Filter q values and calculate next actions
    next_legal_actions_list = []
    for i in range(len(non_final_next_states)):
        mask = next_legal_actions_masks[i]
        legal_indices = [idx for idx, is_legal in enumerate(mask) if is_legal]
        if not legal_indices:
          # If there's no legal actions.  Default to 0.
          next_legal_actions_list.append(0)
        else:
          legal_q_values = next_state_q_online[i, legal_indices]
          best_legal_action_idx = torch.argmax(legal_q_values)
          next_legal_actions_list.append(legal_indices[best_legal_action_idx])
    
    next_actions_tensor = torch.LongTensor(next_legal_actions_list).to(device)

    # Compute the argmax (Double DQN) using target network
    next_q_values_target_temp = target_dqn_model(non_final_next_states_tensor).detach()
    next_q_values_target[non_final_mask] = next_q_values_target_temp.gather(1, next_actions_tensor.unsqueeze(1)).squeeze(1)
    
    expected_q_values = rewards_tensor + (gamma * next_q_values_target)
    
    return q_values, expected_q_values

def optimize_model(optimizer, dqn_model, target_dqn_model, replay_buffer, batch_size, beta, writer, train_steps):
    if len(replay_buffer) < batch_size:
        return

    batch, indices, weights = replay_buffer.sample(batch_size, beta)

    try:
        q_values, expected_q_values = compute_td_errors(batch, dqn_model, target_dqn_model)
    except Exception as e:
        logging.error(f"Error computing TD errors: {str(e)}")
        return
    
    errors = torch.abs(q_values - expected_q_values).detach().cpu().numpy()
    replay_buffer.update_priorities(indices, errors)

    loss = (weights * F.smooth_l1_loss(q_values, expected_q_values, reduction='none')).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dqn_model.parameters(), grad_clip_value)
    optimizer.step()

    writer.add_scalar('Loss/Train', loss.item(), train_steps)

def update_target_network(dqn_model, target_dqn_model):
    target_dqn_model.load_state_dict(dqn_model.state_dict())

def train(env, models, optimizers, replay_buffer, writer, num_episodes=10, batch_size=64, target_update_freq=100, log_freq = 50, test_interval=1000):
    best_eval_score = -float('inf')
    no_improvement_count = 0
    train_steps = 0

    for episode in range(num_episodes):
        betting_history = []
        action_history = []

        for step in range(1000):
            train_steps += 1
            beta = min(1.0, 0.4 + 0.6 * train_steps / 500_000)
            epsilon = max(0.01, 1.0 - 0.99 * train_steps / 500_000)

            trajectories = run_game(env, models, replay_buffer, epsilon, beta, betting_history, action_history)
            
            for optimizer, dqn_model, target_dqn_model in zip(optimizers, models[::2], models[1::2]):  # Update online, skip target
                optimize_model(optimizer, dqn_model, target_dqn_model, replay_buffer, batch_size, beta, writer, train_steps)

            if train_steps % target_update_freq == 0:
                for dqn_model, target_dqn_model in zip(models[::2], models[1::2]):
                    update_target_network(dqn_model, target_dqn_model)

            if train_steps % test_interval == 0:
              eval_score = evaluate_model(env, models, num_eval_episodes=10)
              writer.add_scalar("Evaluation/Average_Return", eval_score, train_steps)

              if eval_score > best_eval_score:
                  best_eval_score = eval_score
                  torch.save(models[0].state_dict(), best_model_path)  # Save best model
                  logging.info(f"New best model saved with score: {best_eval_score}")
                  no_improvement_count = 0
              else:
                no_improvement_count += 1

              logging.info(f"Evaluation score at step {train_steps}: {eval_score}")
                
              if no_improvement_count >= 100:  # early stopping
                logging.info(f"Early stopping triggered at step {train_steps} due to no improvement.")
                return
                
        if episode % log_freq == 0:
            logging.info(f'Episode: {episode}, Buffer Size: {len(replay_buffer)}')

def evaluate_model(env, models, num_eval_episodes=10):
    total_rewards = 0
    env_copy = deepcopy(env)

    for _ in range(num_eval_episodes):
      betting_history = []
      action_history = []
      rewards = eval_episode(env_copy, models, betting_history, action_history)
      total_rewards += rewards[0]  # Assuming we're evaluating the first model

    average_reward = total_rewards / num_eval_episodes
    return average_reward

def eval_episode(env, models, betting_history, action_history):
    trajectories = [[] for _ in range(env.num_players)]
    state, current_player = env.reset()
    pot_size = env.game.pot

    while not env.is_over():
        model = models[current_player]
        state_vector = create_state_vector(env, state, current_player, betting_history, action_history)

        if state_vector is None: return [0] * env.num_players

        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
        legal_actions_mask = get_legal_actions(env, current_player)
        action_probs = get_action_probs(model, state_tensor, legal_actions_mask, training=False)
        masked_probs = np.where(np.array(legal_actions_mask), action_probs.cpu().detach().numpy(), -np.inf)
        action = np.argmax(masked_probs)

        if not legal_actions_mask[action]:
            logging.warning(f"Chosen illegal action {action} during evaluation, choosing a legal action")
            legal_actions = [i for i, legal in enumerate(legal_actions_mask) if legal]
            action = np.random.choice(legal_actions) if legal_actions else 0

        min_bet = get_min_bet(env)
        next_state, next_player, reward, done, next_pot_size, info = step_with_abstraction(env, action, pot_size, min_bet)

        update_history(betting_history, action_history, env, current_player, action, ActionAbstraction().get_bet_amount(action, pot_size, min_bet))
        next_state_vector = create_state_vector(env, next_state, next_player, betting_history, action_history) if not done else None
        trajectories[current_player].append((state_vector, action, reward, next_state_vector, done, legal_actions_mask, current_player))

        state = next_state
        current_player = next_player
        pot_size = next_pot_size

    final_rewards = env.get_payoffs()

    return final_rewards # all rewards

# utils/action_utils.py

import logging
import torch
import numpy as np
import torch.nn.functional as F
from config import device, noise_scale

@torch.jit.script
def fast_normalize(tensor, dim: int = -1):
    return tensor / (torch.linalg.norm(tensor, dim=dim, keepdim=True) + 1e-8)

class ActionAbstraction:
    def __init__(self):
        self.bet_amounts = {
            0: 0,    # FOLD
            1: 0,    # CHECK/CALL
            2: 0.67,  # Bet 2/3 pot
            3: 1,    # Bet pot
            4: 2     # Bet 2x pot
        }
        self.num_actions = len(self.bet_amounts)

    def get_abstract_action(self, action_probs):
        try:
            if not isinstance(action_probs, torch.Tensor):
                action_probs = torch.tensor(action_probs, dtype=torch.float32).to(device)
                
            action_probs_cpu = action_probs.detach().cpu().numpy()
            action = np.random.choice(self.num_actions, p=action_probs_cpu)
            return action
        except Exception as e:
            logging.error(f"Error in get_abstract_action: {str(e)}")
            return 0  # Default to FOLD

    def get_bet_amount(self, abstract_action, pot_size, min_bet):
        if abstract_action not in self.bet_amounts:
            logging.error(f"Invalid abstract action: {abstract_action}")
            return 0
        
        if abstract_action in (0, 1):  # Fold and Check/Call
            return 0
        
        bet_ratio = self.bet_amounts[abstract_action]
        return max(min(int(bet_ratio * pot_size), 300), min_bet)


class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0., theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def get_noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return torch.tensor(self.state, dtype=torch.float).to(device)
    
def get_action_probs(model, state_tensor, legal_actions_mask, training=True):
    try:
        with torch.no_grad():
            q_values = model(state_tensor)
            
            # Softmax on legal actions only
            legal_action_indices = [i for i, is_legal in enumerate(legal_actions_mask) if is_legal]
            if not legal_action_indices:
                return torch.zeros_like(q_values)
                
            legal_q_values = q_values[0, legal_action_indices]
            action_probs = F.softmax(legal_q_values, dim=0)
            
            # Remap the probabilities
            full_probs = torch.zeros_like(q_values).squeeze(0)  # full tensor
            full_probs[legal_action_indices] = action_probs
            
            # Adding noise during training
            if training:
                noise = OrnsteinUhlenbeckNoise(full_probs.numel()).get_noise() * noise_scale
                full_probs += noise
                full_probs = torch.clamp(full_probs, 0.0, 1.0)  # Ensure 0 <= probs <= 1
                full_probs = fast_normalize(full_probs)
                
                
            return full_probs

    except Exception as e:
        logging.error(f"Error in get_action_probs: {str(e)}")
        return torch.zeros(len(legal_actions_mask)).to(device)

def get_legal_actions(env, current_player_id):
    try:
        legal_actions_mask = np.zeros(5, dtype=bool)  # Use a numpy array
        legal_actions = env.get_legal_actions()
        for action in legal_actions:
            legal_actions_mask[action] = True

        return legal_actions_mask.tolist()  # Convert back to list
    except Exception as e:
        logging.error(f"Failed to get legal actions: {str(e)}")
        return [True, False, False, False, False]  # Default, might need better handling


def get_min_bet(env):
    try:
        return env.game.min_raise
    except Exception as e:
        logging.error(f"Failed to get min bet: {str(e)}")
        return 1  # Default value

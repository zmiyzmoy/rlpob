# utils/environment_utils.py
import logging
import rlcard
from rlcard.utils import get_device
import torch
import numpy as np
from .card_utils import extract_cards, cached_evaluate, CardEmbedding
from .action_utils import ActionAbstraction, get_legal_actions, get_min_bet
from config import num_players, device
from collections import deque
from typing import List

def init_env():
    env = rlcard.make(
        'limit-holdem',
        config={
            'game_num_players': num_players,
            'env_id': 'limit-holdem',
            'seed': 42
        }
    )
    return env


def get_state_size(env):
    state_space = env.state_space
    if isinstance(state_space, dict) and 'obs' in state_space:
        return int(np.prod(state_space['obs'].shape))
    else:
        return int(np.prod(state_space.shape))


def _create_stack_vector(env, current_player, max_stack=300):
    """
    Helper for create_state_vector.  Creates the stack size vector.
    """
    stacks_vector = []
    for pid in range(env.num_players):
        try:
            player_id = (current_player + pid) % env.num_players
            player = env.game.players[player_id]
            stack = player.remained_chips if hasattr(player, 'remained_chips') else 0
            stacks_vector.append(stack)
        except Exception as e:
            logging.error(f"Error accessing player {player_id}: {str(e)}")
            stacks_vector.append(0)  # Append a default value
    
    # Normalize and pad/truncate
    stacks_vector = [(s / max_stack) for s in stacks_vector]
    
    while len(stacks_vector) < 6:
        stacks_vector.append(0)
    return stacks_vector[:6]


def _create_betting_history_vector(env, current_player, betting_history):
    """
    Helper for create_state_vector.  Creates the betting history vectors.
    """
    betting_rounds_vector = []
    for round_history in betting_history:
        round_vector = []
        for pid in range(env.num_players):
            player_id = (current_player + pid) % env.num_players
            
            bet_amount = round_history.get(player_id, 0)
            
            if bet_amount > 300:
                logging.warning(f"Bet amount exceeding max, clipping: {bet_amount}")
                bet_amount = 300

            round_vector.append(bet_amount)
            
        while len(round_vector) < 6:
            round_vector.append(0)  # pad/truncate
        betting_rounds_vector.extend(round_vector[:6])  # only take the first 6
    while len(betting_rounds_vector) < 24:  # pad/truncate
        betting_rounds_vector.append(0)
    return betting_rounds_vector[:24]

def _create_action_history_vector(env, current_player, action_history):
    action_history_vector = []
    for round_history in action_history:
        round_vector = []
        for pid in range(env.num_players):
            player_id = (current_player + pid) % env.num_players
            action = round_history.get(player_id, 0)  # Default: FOLD
            round_vector.append(action)
            
        while len(round_vector) < 6:
              round_vector.append(0) # pad/truncate
        action_history_vector.extend(round_vector[:6]) # only take the first 6
    while len(action_history_vector) < 24:  # pad
        action_history_vector.append(0)
    return action_history_vector[:24]

def create_state_vector(env, state, current_player, betting_history, action_history):
    try:
        if not state or 'raw_obs' not in state:
            logging.error("Invalid state format for create_state_vector.")
            return None
            
        raw_obs = state['raw_obs']
        if isinstance(raw_obs, dict) and 'raw_obs' in raw_obs:
                raw_obs = raw_obs['raw_obs']

        # 1. Player Cards Embedding
        player_cards, community_cards = extract_cards(state, env.game.round_name)
        player_card_embedding = CardEmbedding()(player_cards).detach().cpu().numpy()
        community_card_embedding = CardEmbedding()(community_cards).detach().cpu().numpy()

        # 2. Hand Strength
        hand_strength = cached_evaluate(player_cards, community_cards)
        hand_strength_vector = np.array([hand_strength], dtype=np.float32)

        # 3. Pot Size
        pot_size = env.game.pot / 600.0  # Normalize
        pot_size_vector = np.array([pot_size], dtype=np.float32)

        # 4. Stack Sizes
        stacks_vector = _create_stack_vector(env, current_player)

        # 5. Betting History (last 4 rounds)
        betting_rounds_vector = _create_betting_history_vector(env, current_player, betting_history)

        # 6. Action History (last 4 rounds)
        action_history_vector = _create_action_history_vector(env, current_player, action_history)

        # 7. Is last action?
        is_last_vector = np.array([state['raw_obs']['is_last']], dtype=np.float32)
        
        # 8. Current Round
        round_id = env.game.round_counter
        round_one_hot = np.zeros(4)
        if 0 <= round_id < len(round_one_hot):
            round_one_hot[round_id] = 1
        
        final_vector = np.concatenate([
            player_card_embedding,         # 12
            community_card_embedding,      # 12
            hand_strength_vector,          # 1
            pot_size_vector,               # 1
            stacks_vector,                 # 6
            betting_rounds_vector,         # 24
            action_history_vector,         # 24
            is_last_vector,             # 1
            round_one_hot                # 4
        ]).astype(np.float32)              # Total = 85
            
        return final_vector

    except Exception as e:
        logging.error(f"Error in create_state_vector: {str(e)}")
        return None

def update_history(betting_history, action_history, env, current_player_id, action, bet_amount):
    try:
        round_counter = env.game.round_counter
        
        # Make sure there's a list to store betting and action history
        while len(betting_history) <= round_counter:
            betting_history.append({})
        while len(action_history) <= round_counter:
            action_history.append({})
            
        betting_history[round_counter][current_player_id] = betting_history[round_counter].get(current_player_id, 0) + bet_amount
        action_history[round_counter][current_player_id] = action
        
        logging.debug(f"Round {round_counter}: Player {current_player_id} - Bet: {bet_amount}, Action: {action}")
        
    except Exception as e:
        logging.error(f"Error in update_history: {str(e)}")

def step_with_abstraction(env, action, pot_size, min_bet):
    try:
        bet_amount = ActionAbstraction().get_bet_amount(action, pot_size, min_bet)
        next_state, reward, done, info = env.step(action, bet_amount)
        next_player = next_state['raw_obs']['current_player']
        next_pot_size = env.game.pot
            
        return next_state, next_player, reward, done, next_pot_size, info
    except Exception as e:
        logging.error(f"Error in step_with_abstraction: {str(e)}")
        return None, None, 0, True, 0, {}  # Return default values

def run_game(env, models, replay_buffer, epsilon, beta, betting_history, action_history, training=True):
    # Initialize
    trajectories = [[] for _ in range(env.num_players)]
    state, current_player = env.reset()
    
    if not state:
        logging.error("Initial state is None, cannot start game.")
        return []
        
    pot_size = env.game.pot
    
    # Game loop
    while not env.is_over():
        try:
            model = models[current_player]
            state_vector = create_state_vector(env, state, current_player, betting_history, action_history)
            
            if state_vector is None:
                logging.error("Failed to create state vector, aborting run_game")
                return []
                
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
            legal_actions_mask = get_legal_actions(env, current_player)
            action_probs = get_action_probs(model, state_tensor, legal_actions_mask, training)

            if training:
                action = np.random.choice(len(action_probs), p=action_probs.cpu().detach().numpy())
                if not legal_actions_mask[action]:
                    logging.warning(f"Chosen illegal action {action}, choosing a legal action")
                    legal_actions = [i for i, legal in enumerate(legal_actions_mask) if legal]
                    action = np.random.choice(legal_actions) if legal_actions else 0
            else:
                # During evaluation, choose action with highest probability, but still respect legal actions
                masked_probs = np.where(np.array(legal_actions_mask), action_probs.cpu().detach().numpy(), -np.inf)
                action = np.argmax(masked_probs)
                if not legal_actions_mask[action]:
                    logging.warning(f"Chosen illegal action {action} during evaluation, choosing a legal action")
                    legal_actions = [i for i, legal in enumerate(legal_actions_mask) if legal]
                    action = np.random.choice(legal_actions) if legal_actions else 0

            min_bet = get_min_bet(env)  # Get the minimum bet
            next_state, next_player, reward, done, next_pot_size, info = step_with_abstraction(env, action, pot_size, min_bet)
            update_history(betting_history, action_history, env, current_player, action, ActionAbstraction().get_bet_amount(action, pot_size, min_bet))
            next_state_vector = create_state_vector(env, next_state, next_player, betting_history, action_history) if not done else None
            trajectories[current_player].append((state_vector, action, reward, next_state_vector, done, legal_actions_mask, current_player))
            
            state = next_state
            current_player = next_player
            pot_size = next_pot_size
        except Exception as e:
            logging.error(f"Error during game loop: {str(e)}")
            break
            
    # Post-game: Adjust rewards
    final_rewards = env.get_payoffs()
    
    for player_id in range(env.num_players):
        for i, (state_vector, action, reward, next_state_vector, done, legal_actions_mask, recorded_player_id) in enumerate(trajectories[player_id]):
            # Update reward only at the end of the episode, not intermediate steps
            adjusted_reward = final_rewards[recorded_player_id]
            
            replay_buffer.add((state_vector, action, adjusted_reward, next_state_vector, done, legal_actions_mask, recorded_player_id), priority=1.0)
            
    return trajectories  # Returning trajectories might be useful for debugging

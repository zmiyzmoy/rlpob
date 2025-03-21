# utils/card_utils.py

import logging
from treys import Evaluator, Card
from functools import lru_cache
import numpy as np  # Import numpy
import torch
import torch.nn as nn
from config import device


def card_to_number(card_str):
    """Convert a card string (e.g., 'SJ') to a number (0-51)."""
    suits = {'C': 0, 'D': 1, 'H': 2, 'S': 3}
    ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    
    if not isinstance(card_str, str) or len(card_str) < 2:
        logging.error(f"Invalid card format: {card_str}")
        return None
    
    suit = card_str[0].upper()
    rank = card_str[1:]
    if suit not in suits or rank not in ranks:
        logging.error(f"Invalid suit or rank in card {card_str}: suit={suit}, rank={rank}")
        return None
    
    return ranks[rank] + suits[suit] * 13

def number_to_card(card_num):
    """Convert a card number (0-51) back to a string (e.g., 'SJ')."""
    suits = {0: 'C', 1: 'D', 2: 'H', 3: 'S'}
    ranks = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: '10', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}
    
    if not (0 <= card_num <= 51):
        logging.error(f"Invalid card number: {card_num}")
        return None
    
    suit = card_num // 13
    rank = card_num % 13
    return suits[suit] + ranks[rank]

def number_to_treys_card(card_num):
    """Convert a card number (0-51) to treys format (e.g., 'Js' for Jack of Spades)."""
    suits = {0: 'c', 1: 'd', 2: 'h', 3: 's'}
    ranks = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8', 7: '9', 8: 't', 9: 'j', 10: 'q', 11: 'k', 12: 'a'}
    
    if not (0 <= card_num <= 51):
        logging.error(f"Invalid card number for treys conversion: {card_num}")
        return None
    
    suit = card_num // 13
    rank = card_num % 13
    return Card.new(f"{ranks[rank]}{suits[suit]}")

@lru_cache(maxsize=500)
def cached_evaluate(player_cards, community_cards):
    """Evaluate hand strength using treys library."""
    evaluator = Evaluator()
    if not player_cards or len(player_cards) < 2:
        logging.warning(f"Invalid player cards for evaluation: {player_cards}")
        return 0.1
    if not isinstance(community_cards, tuple):
        logging.warning(f"Invalid community cards for evaluation: {community_cards}")
        community_cards = tuple()
    
    try:
        for card in player_cards:
            if not isinstance(card, int) or card < 0 or card > 51:
                logging.error(f"Invalid player card value: {card}")
                return 0.5
        for card in community_cards:
            if not isinstance(card, int) or card < 0 or card > 51:
                logging.error(f"Invalid community card value: {card}")
                return 0.5

        player_treys = [number_to_treys_card(card) for card in player_cards]
        community_treys = [number_to_treys_card(card) for card in community_cards]
        score = evaluator.evaluate(list(community_treys), list(player_treys))
        normalized_score = 1.0 - (score / 7462.0)  # 7462 is the worst possible hand rank
        return normalized_score
    except Exception as e:
        logging.error(f"Error in cached_evaluate: {str(e)}")
        return 0.5

def extract_cards(state_dict, stage):
    """
    Extract and convert player and community cards from the state dictionary.
    Cards are converted to integers in the range 0-51.
    """
    try:
        if 'raw_obs' not in state_dict:
            logging.warning("No 'raw_obs' in state_dict")
            return (), ()
            
        raw_obs = state_dict['raw_obs']
        if isinstance(raw_obs, dict) and 'raw_obs' in raw_obs:
            raw_obs = raw_obs['raw_obs']
            
        player_cards = raw_obs.get('hand', [])
        community_cards = raw_obs.get('public_cards', [])
        
        if not player_cards:
            logging.warning("No player cards found in state")
            return (), ()
        if not isinstance(community_cards, list):
            logging.warning(f"Community cards is not a list: {community_cards}")
            community_cards = []

        logging.debug(f"Raw player_cards: {player_cards}, raw community_cards: {community_cards}")

        player_cards_converted = []
        for card in player_cards:
            card_num = card_to_number(card)
            if card_num is not None:
                player_cards_converted.append(card_num)
            else:
                logging.error(f"Failed to convert player card: {card}")

        community_cards_converted = []
        for card in community_cards:
            card_num = card_to_number(card)
            if card_num is not None:
                community_cards_converted.append(card_num)
            else:
                logging.error(f"Failed to convert community card: {card}")

        logging.debug(f"Converted player_cards: {player_cards_converted}, converted community_cards: {community_cards_converted}")
        return tuple(player_cards_converted), tuple(community_cards_converted)
    except Exception as e:
        logging.error(f"Error in extract_cards: {str(e)}")
        return (), ()
    
class CardEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.rank_embed = nn.Embedding(13, 8)
        self.suit_embed = nn.Embedding(4, 4)
        
    def forward(self, cards):
        try:
            if not cards:
                return torch.zeros(12).to(device)
                
            ranks = []
            suits = []
            for card in cards:
                if not (0 <= card <= 51):
                    logging.error(f"Invalid card index for embedding: {card}")
                    return torch.zeros(12).to(device)
                rank = card % 13  # 0 to 12
                suit = card // 13  # 0 to 3
                ranks.append(rank)
                suits.append(suit)
            
            ranks = torch.tensor(ranks, dtype=torch.long).to(device)
            suits = torch.tensor(suits, dtype=torch.long).to(device)
            
            return torch.cat([
                self.rank_embed(ranks).mean(dim=0),
                self.suit_embed(suits).mean(dim=0)
            ])
        except Exception as e:
            logging.error(f"Card embedding failed: {str(e)}")
            return torch.zeros(12).to(device)

import numpy as np
import pytest
from Monopoly_Go.monopoly_go.env.monopoly_go import MonopolyGoEnv
from Monopoly_Go.monopoly_go.utils import offsets

@pytest.fixture
def dummy_env():
    env = MonopolyGoEnv()
    env.reset()
    return env

def test_fill_action_mask_bankable(dummy_env):
    agent = "player_0"
    dummy_env.state[agent]["hand"] = [0, 1, 2]  # bankable cards
    dummy_env.fill_action_mask(agent)
    mask = dummy_env.infos[agent]["action_mask"]
    assert all(mask[card_id] == 1 for card_id in [0, 1, 2])
    assert mask[-1] == 1  # Do nothing should always be valid

def test_fill_action_mask_pass_go(dummy_env):
    agent = "player_0"
    dummy_env.state[agent]["hand"] = [25]  # Pass Go card
    dummy_env.fill_action_mask(agent)
    mask = dummy_env.infos[agent]["action_mask"]
    assert mask[offsets["pass_go"]] == 1

def test_fill_action_mask_deal_breaker(dummy_env):
    agent = "player_0"
    dummy_env.state[agent]["hand"] = [20]
    dummy_env.state[agent]["other_properties"][0][0] = [1, 2, 3]  # Long enough to mark
    dummy_env.fill_action_mask(agent)
    mask = dummy_env.infos[agent]["action_mask"]
    start = offsets["deal_breaker"]
    assert mask[start + 1] == 1

def test_fill_action_mask_rent_card(dummy_env):
    agent = "player_0"
    dummy_env.state[agent]["hand"] = [49]
    dummy_env.state[agent]["property_slots"][0] = [1]
    dummy_env.fill_action_mask(agent)
    mask = dummy_env.infos[agent]["action_mask"]
    assert np.any(mask[offsets["rent"]:offsets["place_property"]])

def test_fill_action_mask_all_wildcard(dummy_env):
    agent = "player_0"
    dummy_env.state[agent]["hand"] = [99]  # All color wildcard
    dummy_env.fill_action_mask(agent)
    mask = dummy_env.infos[agent]["action_mask"]
    start = offsets["place_wildcard_all"]
    assert all(mask[start + i] == 1 for i in range(10))

def test_fill_action_mask_do_nothing(dummy_env):
    agent = "player_0"
    dummy_env.state[agent]["hand"] = []  # No cards
    dummy_env.fill_action_mask(agent)
    mask = dummy_env.infos[agent]["action_mask"]
    assert mask[-1] == 1  # Do nothing always valid

from Monopoly_Go.monopoly_go import monopoly_go_v0
from Monopoly_Go.monopoly_go.utils import offsets
import numpy as np

def test_sly_deal():
    env = monopoly_go_v0.env(render_mode="human")
    env.reset(seed=42)

    env.state = {
        'player_0': {
            'hand': [np.int64(55), np.int64(89), np.int64(1), np.int64(103)],
            'property_slots': [
                [], [], [], [np.int64(72)], [],
                [np.int64(96)], [], [], [], [np.int64(100)]
            ],
            'bank': [2, 4],
            'other_properties': [
                [[], [], [], [], [], [], [np.int64(80)], [], [], []],
                [[], [], [np.int64(68)], [], [], [], [], [], [], []]
            ],
            'other_banks': [
                [1, 1],
                [1, 5]
            ]
        },

        'player_1': {
            'hand': [
                np.int64(15), np.int64(16), np.int64(57),
                np.int64(39), np.int64(60), np.int64(50), np.int64(81)
            ],
            'property_slots': [
                [], [], [np.int64(69)], [], [], [],
                [np.int64(80)], [], [], []
            ],
            'bank': [1, 1],
            'other_properties': [
                [[], [], [np.int64(68)], [], [], [], [], [], [], []],
                [[], [], [], [np.int64(72)], [], [np.int64(96)], [], [], [], [np.int64(100)]]
            ],
            'other_banks': [
                [1, 5],
                [2, 4]
            ]
        },

        'player_2': {
            'hand': [
                np.int64(71), np.int64(35), np.int64(17),
                np.int64(47), np.int64(56), np.int64(10), np.int64(93)
            ],
            'property_slots': [
                [], [], [np.int64(68)], [], [], [], [], [], [], []
            ],
            'bank': [1, 5],
            'other_properties': [
                [[], [], [], [np.int64(72)], [], [], [], [], [], [np.int64(100)]],
                [[], [], [], [], [], [], [np.int64(80)], [], [], []]
            ],
            'other_banks': [
                [2, 4],
                [1, 1]
            ]
        }
    }

    env.agent_selection = "player_1"
    env.curr_agent_index = 1

    env.step(offsets["sly_deal"] + 2)

    assert env.agent_selection == "player_2"

    env.observe("player_2")
    action_mask = env.infos["player_2"]["action_mask"]

    assert sum(action_mask) == 1
    assert action_mask[offsets["accept_action"]] == 1

    env.step(offsets["accept_action"])

    assert len(env.state["player_1"]["property_slots"][2]) == 2
    assert len(env.state["player_2"]["property_slots"][2]) == 0

    env.observe("player_1")
    assert env.state["player_1"]["other_properties"][0][2] == []

    assert env.agent_selection == "player_1"
    # assert env.current_agent_turns_taken == 1

    env.observe("player_1")

def test_forced_deal():
    env = monopoly_go_v0.env(render_mode="human")
    env.reset(seed=42)

    env.state = {
        'player_0': {
            'hand': [np.int64(35)],  # Forced Deal card
            'property_slots': [
                [], [], [], [], [], [np.int64(70)], [], [], [], []
            ],
            'bank': [],
            'other_properties': [
                [[], [], [np.int64(80)], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        },
        'player_1': {
            'hand': [],
            'property_slots': [
                [], [], [np.int64(80)], [], [], [], [], [], [], []
            ],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [np.int64(70)], [], [], [], []]
            ],
            'other_banks': [[], []]
        },
        'player_2': {
            'hand': [],
            'property_slots': [
                [], [], [], [], [], [], [], [], [], []
            ],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        }
    }

    env.agent_selection = "player_0"
    env.curr_agent_index = 0

    # Step as player_0 to initiate forced deal
    rel = 5 * 20 + 0 * 10 + 2 # offer slot 5, take slot 2 from player_1
    env.step(offsets["forced_deal"] + rel)

    assert env.agent_selection == "player_1"
    env.observe("player_1")

    # Check mask only allows accept
    mask = env.infos["player_1"]["action_mask"]
    assert sum(mask) == 1
    assert mask[offsets["accept_action"]] == 1

    # Player 1 accepts
    env.step(offsets["accept_action"])

    assert env.state["player_0"]["property_slots"][2] == [80]
    assert env.state["player_1"]["property_slots"][5] == [70]

    env.observe("player_0")
    assert env.state["player_0"]["other_properties"][0][5] == [70]
    env.observe("player_1")
    assert env.state["player_1"]["other_properties"][1][2] == [80]


    assert env.agent_selection == "player_0"
    # assert env.current_agent_turns_taken == 1

def test_just_say_no():
    env = monopoly_go_v0.env(render_mode="human")
    env.reset(seed=42)

    initial_state = {
        'player_0': {
            'hand': [np.int64(35)],  # Forced Deal card
            'property_slots': [
                [], [], [], [], [], [np.int64(70)], [], [], [], []
            ],
            'bank': [],
            'other_properties': [
                [[], [], [np.int64(80)], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        },
        'player_1': {
            'hand': [np.int64(24)], # Just say no
            'property_slots': [
                [], [], [np.int64(80)], [], [], [], [], [], [], []
            ],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [np.int64(70)], [], [], [], []]
            ],
            'other_banks': [[], []]
        },
        'player_2': {
            'hand': [],
            'property_slots': [
                [], [], [], [], [], [], [], [], [], []
            ],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        }
    }

    env.state = initial_state
    env.agent_selection = "player_0"
    env.curr_agent_index = 0

    # Step as player_0 to initiate forced deal
    rel = 5 * 20 + 0 * 10 + 2 # offer slot 5, take slot 2 from player_1
    env.step(offsets["forced_deal"] + rel)

    assert env.agent_selection == "player_1"
    env.observe("player_1")

    # Check mask allows accept and just say no
    mask = env.infos["player_1"]["action_mask"]
    assert sum(mask) == 2
    assert mask[offsets["accept_action"]] == 1

    # Player 1 says no
    env.step(offsets["just_say_no"])

    assert env.state == initial_state
    # assert env.current_agent_turns_taken == 1

def test_debt_collector():
    env = monopoly_go_v0.env(render_mode="human")
    env.reset(seed=42)
    bank = [1, 4, 2, 1, 1]
    initial_state = {
        'player_0': {
            'hand': [np.int64(41)],  # Debt Collector card
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        },
        'player_1': {
            'hand': [],
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': bank.copy(),
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        },
        'player_2': {
            'hand': [],
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        }
    }

    env.state = initial_state
    env.agent_selection = "player_0"
    env.curr_agent_index = 0

    # Step as player_0 to initiate debt collection from player_1
    env.step(offsets["debt_collector"])  # Target player_1 with 0 offset

    assert env.agent_selection == "player_1"
    env.observe("player_1")

    # Player_1 should  accept
    mask = env.infos["player_1"]["action_mask"]
    assert sum(mask) == 1
    assert mask[offsets["accept_action"]] == 1

    # Player 1 accepts
    env.step(offsets["accept_action"])

    assert sum(env.state["player_1"]["bank"]) == sum(bank) - 5
    assert sum(env.state["player_0"]["bank"]) == 5

    assert env.agent_selection == "player_0"

def test_rent_payment():
    env = monopoly_go_v0.env(render_mode="human")
    env.reset(seed=42)

    bank = [1, 2, 2]  # Total = 5

    initial_state = {
        'player_0': {
            'hand': [np.int64(49)],  # Rent Blue/Green #1
            'property_slots': [
                [np.int64(63)], [], [], [], [], [], [], [], [], []
            ],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], bank.copy()]
        },
        'player_1': {
            'hand': [],
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': bank.copy(),
            'other_properties': [
                [[np.int64(63)], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        },
        'player_2': {
            'hand': [],
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        }
    }

    env.state = initial_state
    env.agent_selection = "player_0"
    env.curr_agent_index = 0

    # Rent offset: 0 → player_1, card_49 (Blue/Green #1), not doubled
    rent_offset = offsets["rent"] + 0
    env.step(rent_offset)

    assert env.agent_selection == "player_1"
    env.observe("player_1")

    mask = env.infos["player_1"]["action_mask"]
    assert sum(mask) == 1
    assert mask[offsets["accept_action"]] == 1

    # Player 1 pays the rent
    env.step(offsets["accept_action"])

    assert sum(env.state["player_1"]["bank"]) == sum(bank) - 3
    assert sum(env.state["player_0"]["bank"]) == 3
    assert env.agent_selection == "player_0"

def test_birthday():
    env = monopoly_go_v0.env(render_mode="human")
    env.reset(seed=42)

    p1_bank = [1, 2, 2]  # Total = 5
    p2_bank = [4]

    initial_state = {
        'player_0': {
            'hand': [np.int64(45)],
            'property_slots': [
                [np.int64(63)], [], [], [], [], [], [], [], [], []
            ],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [p1_bank.copy(), p2_bank.copy()]
        },
        'player_1': {
            'hand': [],
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': p1_bank.copy(),
            'other_properties': [
                [[np.int64(63)], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [p2_bank.copy(), []]
        },
        'player_2': {
            'hand': [],
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': p2_bank.copy(),
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], [p1_bank.copy()]]
        }
    }

    env.state = initial_state
    env.agent_selection = "player_0"
    env.curr_agent_index = 0

    env.step(offsets["birthday"])

    assert env.agent_selection == "player_1"
    env.observe("player_1")

    mask = env.infos["player_1"]["action_mask"]
    assert sum(mask) == 1
    assert mask[offsets["accept_action"]] == 1

    # Player 1 pays the rent
    env.step(offsets["accept_action"])
    assert env.action_completed == False

    assert env.agent_selection == "player_2"
    env.observe("player_2")

    mask = env.infos["player_2"]["action_mask"]
    assert sum(mask) == 1
    assert mask[offsets["accept_action"]] == 1

    # Player 2 pays the rent
    env.step(offsets["accept_action"])
    assert env.action_completed == True

    assert sum(env.state["player_1"]["bank"]) == sum(p1_bank) - 2
    assert sum(env.state["player_2"]["bank"]) == 0

    assert env.agent_selection == "player_0"

def test_pay_from_property():
    env = monopoly_go_v0.env(render_mode="human")
    env.reset(seed=42)

    # Player 1 has no bank but one property card worth 2
    p1_props = [[np.int64(63)], [], [], [], [], [], [], [], [], []]  # Assume 63 is worth 2
    initiator_bank = []

    env.state = {
        'player_0': {
            'hand': [np.int64(41)],  # Debt collector
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': initiator_bank.copy(),
            'other_properties': [p1_props.copy(), [[]]],
            'other_banks': [[], []]
        },
        'player_1': {
            'hand': [],
            'property_slots': p1_props.copy(),
            'bank': [],  # Empty bank
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        },
        'player_2': {
            'hand': [],
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                p1_props.copy()
            ],
            'other_banks': [[], []]
        }
    }

    env.agent_selection = "player_0"
    env.curr_agent_index = 0

    # Initiate debt collection from player_1
    env.step(offsets["debt_collector"])  # Target player_1

    # player_1 should have option to pay from properties
    assert env.agent_selection == "player_1"
    env.observe("player_1")

    env.step(offsets["accept_action"])
    assert env.agent_selection == "player_1"
    env.observe("player_1")
    assert env.in_action_cycle == True
    action_mask = env.infos["player_1"]["action_mask"]
    assert action_mask[offsets["pay_sum_from_property"]] == 1

    env.step(offsets["pay_sum_from_property"])
    assert sum(env.state["player_1"]["bank"]) == 0
    assert all(len(slot) == 0 for slot in env.state["player_1"]["property_slots"])
    assert env.state["player_0"]["property_slots"][0] == [np.int64(63)]  # Property value
    assert env.agent_selection == "player_0"

def test_pay_from_property_bank_and_property():
    env = monopoly_go_v0.env(render_mode="human")
    env.reset(seed=42)

    p1_props = [[np.int64(63)], [], [], [], [], [], [], [], [], []]  # Assume 63 is worth 2
    initiator_bank = []

    env.state = {
        'player_0': {
            'hand': [np.int64(41)],  # Debt collector
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': initiator_bank.copy(),
            'other_properties': [p1_props.copy(), [[]]],
            'other_banks': [[], [1]]
        },
        'player_1': {
            'hand': [],
            'property_slots': p1_props.copy(),
            'bank': [1],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                [[], [], [], [], [], [], [], [], [], []]
            ],
            'other_banks': [[], []]
        },
        'player_2': {
            'hand': [],
            'property_slots': [[], [], [], [], [], [], [], [], [], []],
            'bank': [],
            'other_properties': [
                [[], [], [], [], [], [], [], [], [], []],
                p1_props.copy()
            ],
            'other_banks': [[], []]
        }
    }

    env.agent_selection = "player_0"
    env.curr_agent_index = 0

    # Initiate debt collection from player_1
    env.step(offsets["debt_collector"])  # Target player_1

    # player_1 should have option to pay from properties
    assert env.agent_selection == "player_1"
    env.observe("player_1")

    env.step(offsets["accept_action"])
    assert env.agent_selection == "player_1"

    env.observe("player_1")
    assert sum(env.state["player_1"]["bank"]) == 0
    action_mask = env.infos["player_1"]["action_mask"]
    assert action_mask[offsets["pay_sum_from_property"]] == 1

    env.step(offsets["pay_sum_from_property"])
    assert all(len(slot) == 0 for slot in env.state["player_1"]["property_slots"])
    assert env.state["player_0"]["property_slots"][0] == [np.int64(63)]  # Property value
    assert env.agent_selection == "player_0"

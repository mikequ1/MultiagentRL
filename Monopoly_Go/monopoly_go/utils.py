import logging
from itertools import combinations

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MONOPOLY_DEAL_CARDS = {
    # Money Cards
    0: "Money 1M #1",
    1: "Money 1M #2",
    2: "Money 1M #3",
    3: "Money 1M #4",
    4: "Money 1M #5",
    5: "Money 1M #6",
    6: "Money 2M #1",
    7: "Money 2M #2",
    8: "Money 2M #3",
    9: "Money 2M #4",
    10: "Money 2M #5",
    11: "Money 3M #1",
    12: "Money 3M #2",
    13: "Money 3M #3",
    14: "Money 4M #1",
    15: "Money 4M #2",
    16: "Money 4M #3",
    17: "Money 5M #1",
    18: "Money 5M #2",
    19: "Money 10M",

    # Action Cards
    20: "Deal Breaker #1",
    21: "Deal Breaker #2",
    22: "Just Say No #1",
    23: "Just Say No #2",
    24: "Just Say No #3",
    25: "Pass Go #1",
    26: "Pass Go #2",
    27: "Pass Go #3",
    28: "Pass Go #4",
    29: "Pass Go #5",
    30: "Pass Go #6",
    31: "Pass Go #7",
    32: "Pass Go #8",
    33: "Pass Go #9",
    34: "Pass Go #10",
    35: "Forced Deal #1",
    36: "Forced Deal #2",
    37: "Forced Deal #3",
    38: "Sly Deal #1",
    39: "Sly Deal #2",
    40: "Sly Deal #3",
    41: "Debt Collector #1",
    42: "Debt Collector #2",
    43: "Debt Collector #3",
    44: "It's My Birthday #1",
    45: "It's My Birthday #2",
    46: "It's My Birthday #3",

    # Double the Rent
    47: "Double The Rent #1",
    48: "Double The Rent #2",

    # Rent Cards
    49: "Rent Blue/Green #1",
    50: "Rent Blue/Green #2",
    51: "Rent Red/Yellow #1",
    52: "Rent Red/Yellow #2",
    53: "Rent Pink/Orange #1",
    54: "Rent Pink/Orange #2",
    55: "Rent Light Blue/Brown #1",
    56: "Rent Light Blue/Brown #2",
    57: "Rent Railroad/Utility #1",
    58: "Rent Railroad/Utility #2",
    59: "Rent Wild #1",
    60: "Rent Wild #2",
    61: "Rent Wild #3",

    # Property Cards
    62: "Property Blue #1",
    63: "Property Blue #2",

    64: "Property Green #1",
    65: "Property Green #2",
    66: "Property Green #3",

    67: "Property Red #1",
    68: "Property Red #2",
    69: "Property Red #3",

    70: "Property Yellow #1",
    71: "Property Yellow #2",
    72: "Property Yellow #3",

    73: "Property Pink #1",
    74: "Property Pink #2",
    75: "Property Pink #3",

    76: "Property Orange #1",
    77: "Property Orange #2",
    78: "Property Orange #3",

    79: "Property Light Blue #1",
    80: "Property Light Blue #2",
    81: "Property Light Blue #3",

    82: "Property Railroad #1",
    83: "Property Railroad #2",
    84: "Property Railroad #3",
    85: "Property Railroad #4",

    86: "Property Utility #1",
    87: "Property Utility #2",

    88: "Property Brown #1",
    89: "Property Brown #2",

    # Wildcards
    90: "Wildcard Blue/Green",
    91: "Wildcard Green/Railroad",
    92: "Wildcard Utility/Railroad",
    93: "Wildcard Light Blue/Railroad",
    94: "Wildcard Light Blue/Brown",
    95: "Wildcard Pink/Orange #1",
    96: "Wildcard Pink/Orange #2",
    97: "Wildcard Red/Yellow #1",
    98: "Wildcard Red/Yellow #2",
    99: "Wildcard All Colors #1",
    100: "Wildcard All Colors #2",

    # House and Hotel
    101: "House #1",
    102: "House #2",
    103: "House #3",
    104: "Hotel #1",
    105: "Hotel #2",
}

CARD_VALUES = {
    # Money Cards
    0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1,  # Money 1M
    6: 2, 7: 2, 8: 2, 9: 2, 10: 2,       # Money 2M
    11: 3, 12: 3, 13: 3,                 # Money 3M
    14: 4, 15: 4, 16: 4,                 # Money 4M
    17: 5, 18: 5,                        # Money 5M
    19: 10,                              # Money 10M

    # Action Cards
    20: 5, 21: 5,                        # Deal Breaker
    22: 4, 23: 4, 24: 4,                 # Just Say No
    25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1,  # Pass Go
    35: 3, 36: 3, 37: 3,                 # Forced Deal
    38: 3, 39: 3, 40: 3,                 # Sly Deal
    41: 3, 42: 3, 43: 3,                 # Debt Collector
    44: 2, 45: 2, 46: 2,                 # It's My Birthday
    47: 1, 48: 1,                        # Double The Rent

    # Rent Cards
    49: 1, 50: 1,                        # Rent Blue/Green
    51: 1, 52: 1,                        # Rent Red/Yellow
    53: 1, 54: 1,                        # Rent Pink/Orange
    55: 1, 56: 1,                        # Rent Light Blue/Brown
    57: 1, 58: 1,                        # Rent Railroad/Utility
    59: 3, 60: 3, 61: 3,                 # Rent Wild

    # Property Cards
    62: 4, 63: 4,                        # Property Blue
    64: 1, 65: 1,                        # Property Brown
    66: 2, 67: 2,                        # Property Utility
    68: 4, 69: 4, 70: 4,                 # Property Green
    71: 3, 72: 3, 73: 3,                 # Property Yellow
    74: 3, 75: 3, 76: 3,                 # Property Red
    77: 2, 78: 2, 79: 2,                 # Property Orange
    80: 2, 81: 2, 82: 2,                 # Property Pink
    83: 1, 84: 1, 85: 1,                 # Property Light Blue
    86: 2, 87: 2, 88: 2, 89: 2,          # Property Railroad

    # Wildcards
    90: 4,                               # Wildcard Blue/Green
    91: 4,                               # Wildcard Green/Railroad
    92: 2,                               # Wildcard Utility/Railroad
    93: 1,                               # Wildcard Light Blue/Railroad
    94: 1,                               # Wildcard Light Blue/Brown
    95: 2, 96: 2,                        # Wildcard Pink/Orange
    97: 3, 98: 3,                        # Wildcard Red/Yellow
    99: 0, 100: 0,                       # Wildcard All Colors

    # House and Hotel
    101: 3, 102: 3, 103: 3,              # House
    104: 4, 105: 4                       # Hotel
}

WILDCARD_TO_PROPERTY_SLOTS = {
    90: (0, 1),
    91: (1, 7),
    92: (8, 7),
    93: (6, 7),
    94: (6, 9),
    95: (4, 5),
    96: (4, 5),
    97: (2, 3),
    98: (2, 3),
}

PROPERTY_TYPE_INDEX = {62: 0, 63: 0, 64: 1, 65: 1, 66: 1, 67: 2, 68: 2, 69: 2, 70: 3, 71: 3, 72: 3, 73: 4,
                       74: 4, 75: 4, 76: 5, 77: 5, 78: 5, 79: 6, 80: 6, 81: 6, 82: 7, 83: 7, 84: 7, 85: 7,
                       86: 8, 87: 8, 88: 9, 89: 9}

CARDS_IN_PROPERTY_SLOTS = (
    (62, 63),
    (64, 65, 66),
    (67, 68, 69),
    (70, 71, 72),
    (73, 74, 75),
    (76, 77, 78),
    (79, 80, 81),
    (82, 83, 84, 85),
    (86, 87),
    (88, 89)
)

RENT_VALUES = [
    [3, 8],               # 1 property: 3M, 2 properties: 8M
    [2, 4, 7],           # 1: 2M, 2: 4M, 3: 7M
    [2, 3, 6],             # 1: 2M, 2: 3M, 3: 6M
    [2, 4, 6],          # 1: 2M, 2: 4M, 3: 6M
    [1, 2, 4],            # 1: 1M, 2: 2M, 3: 4M
    [1, 3, 5],          # 1: 1M, 2: 3M, 3: 5M
    [1, 2, 3],      # 1: 1M, 2: 2M, 3: 3M
    [1, 2],              # 1: 1M, 2: 2M
    [1, 2, 3, 4],     # 1: 1M, 2: 2M, 3: 3M, 4: 4M
    [1, 2]             # 1: 1M, 2: 2M
]

CARDS_TO_COMPLETE = (2, 3, 3, 3, 3, 3, 3, 4, 2, 2)

num_players = 3
NUM_RENT_CARDS = 5
NUM_PROPERTY_CARDS = 10
NUM_PROPERTY_SLOTS = 10
OTHER_PLAYERS_PROPERTY_SLOTS = (num_players-1) * 10
NUM_CARDS = len(MONOPOLY_DEAL_CARDS)
BANKABLE_CARDS = NUM_CARDS - 39
ACTION_SPACE_LENGTH = (
    BANKABLE_CARDS +
    OTHER_PLAYERS_PROPERTY_SLOTS + # Player + property slot for deal breaker
    1 + # Just Say No
    1 + # Pass Go
    OTHER_PLAYERS_PROPERTY_SLOTS + # Sly deal
    10 * (num_players - 1) * 10 + # Forced Deal (Card to trade in own property slot and card to swap with)
    (num_players - 1) + # Debt collector
    1 + # It's my birthday
    NUM_RENT_CARDS * 2 * (num_players-1) + # Distinct rent cards + double the rent or not
    NUM_PROPERTY_SLOTS + # Place property
    NUM_PROPERTY_SLOTS + # Place wildcard (except for wildcard all colors)
    NUM_PROPERTY_SLOTS + # Flip wildcard (in property slot)
    NUM_PROPERTY_SLOTS + # Place wildcard all colors
    NUM_PROPERTY_SLOTS * NUM_PROPERTY_SLOTS + # Move wildcard all
    NUM_PROPERTY_SLOTS + # Place House
    NUM_PROPERTY_SLOTS + # Place Hotel
    NUM_PROPERTY_SLOTS + # Pay sum with property
    1 + # Accept deal
    NUM_CARDS + # Discard
    2 # Do nothing
)

# Define constants first in a centralized layout
def build_action_offsets(BANKABLE_CARDS, OTHER_PLAYERS_PROPERTY_SLOTS, NUM_RENT_CARDS, NUM_PROPERTY_SLOTS, num_players):
    offsets = {}
    cursor = BANKABLE_CARDS

    offsets["deal_breaker"] = cursor
    cursor += OTHER_PLAYERS_PROPERTY_SLOTS  # Slot per player

    offsets["just_say_no"] = cursor
    cursor += 1

    offsets["pass_go"] = cursor
    cursor += 1

    offsets["sly_deal"] = cursor
    cursor += OTHER_PLAYERS_PROPERTY_SLOTS

    offsets["forced_deal"] = cursor
    cursor += 10 * (num_players-1) * 10

    offsets["debt_collector"] = cursor
    cursor += (num_players - 1) + 1  # One bit per player + extra

    offsets["birthday"] = cursor
    cursor += 1

    offsets["rent"] = cursor
    cursor += NUM_RENT_CARDS * 2 * (num_players - 1)

    offsets["place_property"] = cursor
    cursor += NUM_PROPERTY_SLOTS

    offsets["place_wildcard"] = cursor
    cursor += NUM_PROPERTY_SLOTS

    offsets["flip_wildcard"] = cursor
    cursor += NUM_PROPERTY_SLOTS

    offsets["place_wildcard_all"] = cursor
    cursor += NUM_PROPERTY_SLOTS

    offsets["move_wildcard_all"] = cursor
    cursor += NUM_PROPERTY_SLOTS * NUM_PROPERTY_SLOTS

    offsets["place_house"] = cursor
    cursor += NUM_PROPERTY_SLOTS

    offsets["place_hotel"] = cursor
    cursor += NUM_PROPERTY_SLOTS

    offsets["pay_sum_from_property"] = cursor
    cursor += NUM_PROPERTY_SLOTS

    offsets["accept_action"] = cursor
    cursor += 1

    offsets["discard_card"] = cursor
    cursor += NUM_CARDS

    offsets["do_nothing"] = cursor
    cursor += 1

    return offsets

offsets = build_action_offsets(
        BANKABLE_CARDS,
        OTHER_PLAYERS_PROPERTY_SLOTS,
        NUM_RENT_CARDS,
        NUM_PROPERTY_SLOTS,
        num_players
)
# Return action, target_player, property_slot
def get_action(action):

    if action < offsets["deal_breaker"]:
        return "bank", -1, -1

    elif action < offsets["just_say_no"]:
        rel = action - offsets["deal_breaker"]
        return "deal_breaker", rel // 10, rel % 10

    elif action == offsets["just_say_no"]:
        return "just_say_no", -1, -1

    elif action == offsets["pass_go"]:
        return "pass_go", -1, -1

    elif offsets["sly_deal"] <= action < offsets["forced_deal"]:
        rel = action - offsets["sly_deal"]
        return "sly_deal", rel // 10 , rel % 10

    elif offsets["forced_deal"] <= action < offsets["debt_collector"]:
        rel = action - offsets["forced_deal"]

        your_slot = rel // ((num_players - 1) * 10)
        opponent_idx = (rel % ((num_players - 1) * 10)) // 10
        opponent_slot = rel % 10

        return "forced_deal", opponent_idx, (your_slot, opponent_slot)

    elif action in (offsets["debt_collector"], offsets["debt_collector"] + 1):
        return "debt_collector", action - offsets["debt_collector"], -1

    elif action == offsets["birthday"]:
        return "birthday", -1, -1

    elif offsets["rent"] <= action < offsets["place_property"]:
        rel = action - offsets["rent"]
        target_player = rel // (NUM_RENT_CARDS * 2)
        within_player = rel % (NUM_RENT_CARDS * 2)
        rent_offset = within_player // 2
        double = within_player % 2 == 1
        return "rent", target_player, (rent_offset, double)

    elif offsets["place_property"] <= action < offsets["place_wildcard"]:
        return "place_property", -1, action - offsets["place_property"]

    elif offsets["place_wildcard"] <= action < offsets["flip_wildcard"]:
        rel = action - offsets["place_wildcard"]
        return "place_wildcard", -1, rel

    elif offsets["flip_wildcard"] <= action < offsets["place_wildcard_all"]:
        return "flip_wildcard", -1, action - offsets["flip_wildcard"]

    elif offsets["place_wildcard_all"] <= action < offsets["move_wildcard_all"]:
        return "place_wildcard_all", -1, action - offsets["place_wildcard_all"]

    elif offsets["move_wildcard_all"] <= action < offsets["place_house"]:
        rel = action - offsets["move_wildcard_all"]
        src = rel // NUM_PROPERTY_SLOTS
        dst = rel % NUM_PROPERTY_SLOTS
        return "move_wildcard_all", -1, (src, dst)

    elif action == offsets["place_house"]:
        return "place_house", -1, action - offsets["place_house"]

    elif action == offsets["place_hotel"]:
        return "place_hotel", -1, action - offsets["place_hotel"]

    elif offsets["pay_sum_from_property"] <= action < offsets["pay_sum_from_property"] + NUM_PROPERTY_SLOTS:
        return "pay_sum_from_property", -1, action - offsets["pay_sum_from_property"]

    elif action == offsets["accept_action"]:
        return "accept_action", -1, -1

    elif offsets["discard_card"] <= action < offsets["do_nothing"]:
        return "discard_card", -1, action - offsets["discard_card"]

    elif action == offsets["do_nothing"]:
        return "do_nothing", -1, -1

    else:
        raise ValueError(f"Unrecognized action index: {action}")


def pay_from_bank(bank: list[int], amount: int) -> list[int]:
    from collections import defaultdict

    n = len(bank)
    # dp[total] = list of indices used to reach this total
    dp = defaultdict(list)
    dp[0] = []

    for i, val in enumerate(bank):
        current_dp = dict(dp)  # snapshot to avoid in-place updates
        for total, indices in current_dp.items():
            new_total = total + val
            if new_total not in dp or len(dp[new_total]) > len(indices) + 1:
                dp[new_total] = indices + [i]

    # Find the smallest total >= amount
    valid_totals = [t for t in dp if t >= amount]
    if not valid_totals:
        return []

    best_total = min(valid_totals, key=lambda t: (t, len(dp[t])))
    best_indices = dp[best_total]

    paid_values = [bank[i] for i in best_indices]
    for i in sorted(best_indices, reverse=True):
        bank.pop(i)

    return paid_values

flattened_length = (12 + 50 + (8*NUM_PROPERTY_SLOTS) +
                    (8 * NUM_PROPERTY_SLOTS * (num_players - 1)) +
                    (50 * (num_players - 1)))
from collections import deque
import logging
from pprint import pprint

import numpy as np
from gymnasium.spaces import Dict, Discrete, Box
from gymnasium.utils import seeding
from pettingzoo import AECEnv

from Monopoly_Go.monopoly_go.utils import (
    ACTION_SPACE_LENGTH,
    CARDS_IN_PROPERTY_SLOTS,
    CARDS_TO_COMPLETE,
    CARD_VALUES,
    NUM_PROPERTY_SLOTS,
    NUM_RENT_CARDS,
    PROPERTY_TYPE_INDEX,
    RENT_VALUES,
    WILDCARD_TO_PROPERTY_SLOTS,

    get_action,
    offsets,
    pay_from_bank,
)

NUM_ACTION_CARDS = (29 + 23)
MAX_HAND_SIZE=10
MAX_PROPERTY_LENGTH = 10

MONEY_RANGE = set([n for n in range(20)])
ACTION_RANGE = set([n for n in range(20, 62)])
PROPERTY_RANGE = set([n for n in range(62, 90)])
WILDCARD_RANGE = set([n for n in range(90, 101)])
HOUSE_HOTEL_RANGE = set(n for n in range(101, 106))

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

place_property_reward = 5
complete_set_reward = 20
accept_action_reward = -5
deck_run_out_reward = -20
discard_reward = -10
rare_action_reward = 0
common_action_reward = 0
bank_reward = -100
win_reward = 60
loss_reward = -60

class MonopolyGoEnv(AECEnv):

    metadata = {
        "name": "monopoly_go_v0"
    }

    def __init__(self):
        super().__init__()

        self.deck = np.arange(105)
        np.random.shuffle(self.deck)
        self.deck = deque(self.deck)
        self.possible_agents = ["player_" + str(r) for r in range(3)]
        self.discard_pile = deque()
        self.winner = -1

        self.action_counts = {a: {} for a in self.possible_agents}

    def action_space(self, agent=None):
        return Discrete(ACTION_SPACE_LENGTH)

    def observation_space(self, agent=None):
        pass

    def draw_from_deck(self):
        if not self.deck:
            if not self.discard_pile:
                print("Game over due to deck running out")
                self.deck_ran_out = True
                return -1
            self.deck = self.discard_pile
            np.random.shuffle(self.deck)
            self.discard_pile = []

        return self.deck.pop()


    def fill_action_mask(self, agent):
        action_mask = np.zeros(ACTION_SPACE_LENGTH, dtype=np.int8)

        # If hand is empty, do nothing
        if not self.state[agent]["hand"] and not self.in_action_cycle and not self.pay_from_property:
            action_mask[-1] = 1
            self.infos[agent]["action_mask"] = action_mask
            return

        def mark_deal_breaker():
            start = offsets["deal_breaker"]
            for i, properties in enumerate(self.state[agent]["other_properties"]):
                for property_slot in properties:
                    if len(property_slot) > 2:
                        logger.info(f"Deal breaker: marking {start + 1}")
                        action_mask[start + 1] = 1

        def mark_sly_deal():
            start = offsets["sly_deal"]
            for j, properties in enumerate(self.state[agent]["other_properties"]):
                for i, property_slot in enumerate(properties):
                    if len(property_slot) > 0:
                        action_mask[start+(j*10)+i] = 1

        def mark_just_say_no():
            start = offsets["just_say_no"]
            for card_id in hand:
                if card_id in (22,23,24):
                    action_mask[start] = 1
                    break

        def mark_pass_go():
            start = offsets["pass_go"]
            logger.info(f"Pass go: marking {start}")
            action_mask[start] = 1

        def mark_forced_deal():
            start = offsets["forced_deal"]
            own_properties = self.state[agent]["property_slots"]
            other_properties = self.state[agent]["other_properties"]

            for your_slot_idx, your_slot in enumerate(own_properties):
                if len(your_slot) == 0:
                    continue

                for opp_idx, opp_property_slots in enumerate(other_properties):  # opponent index
                    for opp_slot_idx, opp_slot in enumerate(opp_property_slots):
                        if len(opp_slot) == 0:
                            continue

                        rel = (
                                your_slot_idx * (len(other_properties) * 10) +
                                opp_idx * 10 +
                                opp_slot_idx
                        )
                        index = start + rel
                        logger.info(
                            f"Forced deal: marking index {index} for your slot {your_slot_idx}, opponent {opp_idx}, their slot {opp_slot_idx}")
                        action_mask[index] = 1

        def mark_debt_collector():
            start = offsets["debt_collector"]
            action_mask[start] = 1
            action_mask[start+1] = 1

        def mark_birthday():
            start = offsets["birthday"]
            action_mask[start] = 1

        def mark_rent(card_id, has_double_the_rent: bool):
            start = offsets["rent"]
            property_slots = self.state[agent]["property_slots"]

            rent_card_index = (card_id - 49) - ((card_id - 49) % 2)
            logger.info(f"Current hand: {hand}")
            if len(property_slots[rent_card_index//2]) > 0 or len(property_slots[rent_card_index//2+1]) > 0:
                if not any(
                        property_slots[rent_card_index//2 + i][0] not in (99, 100)
                        for i in range(2) if len(property_slots[rent_card_index//2+i]) > 0
                ):
                    return
                logger.info(f"Properties: {property_slots}")
                action_mask[start + rent_card_index] = 1
                action_mask[start + rent_card_index + 10] = 1
                if has_double_the_rent:
                    action_mask[start + rent_card_index + 1] = 1
                    action_mask[start + rent_card_index + 10 + 1] = 1
            else:
                logger.info("Does not have associated property card")

        def mark_property(card_id):
            start = offsets["place_property"]
            property_slot_index = PROPERTY_TYPE_INDEX[card_id]
            logger.info(f"Property slot index: {property_slot_index}")
            if len(self.state[agent]["property_slots"][property_slot_index]) < 3:
                logger.info(f"Property: marking {start + property_slot_index}")
                action_mask[start + property_slot_index] = 1

        def mark_colored_wildcard(card_id):
            start = offsets["place_wildcard"]

            a, b = WILDCARD_TO_PROPERTY_SLOTS[card_id]
            logger.info(f"Wildcard {card_id} eligible for slots {a} and {b}")
            if len(self.state[agent]["property_slots"][a]) == 0:
                action_mask[start + a] = 1
            if len(self.state[agent]["property_slots"][b]) == 0:
                action_mask[start + b] = 1

        def mark_flip_wildcard(property_slot: int):
            start = offsets["flip_wildcard"]
            action_mask[start + property_slot] = 1

        def mark_all_wildcard():
            start = offsets["place_wildcard_all"]
            logger.info(f"All wildcard: marking {start} to {start+9}")
            for i in range(10):
                action_mask[start + i] = 1

        def mark_move_wildcard(property_slot: int):
            start = offsets["move_wildcard_all"]
            for i in range(10):
                action_mask[start + property_slot*10 + i] = 1

        def mark_house():
            start = offsets["place_house"]
            for i, property_slot in enumerate(self.state[agent]["property_slots"]):
                if len(property_slot) > 2:
                    action_mask[start] = 1

        def mark_hotel():
            start = offsets["place_hotel"]
            for i, property_slot in enumerate(self.state[agent]["property_slots"]):
                if len(property_slot) > 2:
                    action_mask[start] = 1

        def mark_pay_sum_with_property():
            start = offsets["pay_sum_from_property"]
            for i, property_slot in enumerate(self.state[agent]["property_slots"]):
                if len(property_slot) > 0:
                    action_mask[start + i] = 1

        def mark_accept_action():
            start = offsets["accept_action"]
            action_mask[start] = 1

        hand = self.state[agent]["hand"]
        if self.must_discard:
            for card_id in hand:
                action_mask[offsets["discard_card"] + card_id] = 1
            self.infos[agent]["action_mask"] = action_mask
            return

        if self.in_action_cycle:
            if self.pay_from_property:
                mark_pay_sum_with_property()
                self.infos[agent]["action_mask"] = action_mask
                return
            mark_just_say_no()
            mark_accept_action()
            action_mask[-1] = 0
            self.infos[agent]["action_mask"] = action_mask
            return

        has_double_the_rent = 47 in hand or 48 in hand

        for card_id in hand:
            if card_id in (47, 48):
                continue
            if card_id < 62 or card_id > 100:
                action_mask[card_id if card_id < 62 else card_id - 39] = 1

            if card_id in (20, 21):
                logger.info(f"Card ID {card_id} indicates deal breaker. Calling mark_deal_breaker()")
                mark_deal_breaker()
            elif card_id in (38, 39, 40):
                logger.info(f"Card ID {card_id} indicates sly deal. Calling mark_sly_deal()")
                mark_sly_deal()
            elif 25 <= card_id <= 34:
                logger.info(f"Card ID {card_id} indicates pass go. Calling mark_pass_go()")
                mark_pass_go()
            elif card_id in (35, 36, 37):
                logger.info(f"Card ID {card_id} indicates forced deal. Calling mark_forced_deal()")
                mark_forced_deal()
            elif card_id in (41, 42, 43):
                logger.info(f"Card ID {card_id} indicates debt collector. Calling mark_debt_collector()")
                mark_debt_collector()
            elif card_id in (44, 45, 46):
                logger.info(f"Card ID {card_id} indicates birthday. Calling mark_birthday()")
                mark_birthday()
            elif 49 <= card_id <= 58:
                logger.info(f"Card ID {card_id} is a rent card. Calling mark_rent()")
                mark_rent(card_id, has_double_the_rent=has_double_the_rent)
            elif card_id in (59, 60, 61):
                logger.info(f"Card ID {card_id} is a wild rent card. Calling mark_rent() for all rent cards")
                for i in range(NUM_RENT_CARDS * 2):
                    mark_rent(49 + i , has_double_the_rent=has_double_the_rent)
            elif 62 <= card_id <= 89:
                logger.info(f"Card ID {card_id} is a property card. Calling mark_property()")
                mark_property(card_id)
            elif 90 <= card_id <= 98:
                logger.info(f"Card ID {card_id} is a colored wildcard. Calling mark_colored_wildcard()")
                mark_colored_wildcard(card_id)
            elif card_id in (99, 100):
                logger.info(f"Card ID {card_id} is a wildcard all. Calling mark_all_wildcard()")
                mark_all_wildcard()
            elif card_id in (101, 102, 103):
                logger.info(f"Card ID {card_id} is a house. Calling mark_house()")
                mark_house()
            elif card_id in (104, 105):
                logger.info(f"Card ID {card_id} is a hotel. Calling mark_hotel()")
                mark_hotel()

        for i, property_slot in enumerate(self.state[agent]["property_slots"]):
            for card_id in property_slot:
                if 90 <= card_id <= 98:
                    mark_flip_wildcard(i)
                if card_id in (99, 100):
                    mark_move_wildcard(i)

        action_mask[-1] = 1
        self.infos[agent]["action_mask"] = action_mask
        return action_mask

    def render(self):
        pass

    def observe(self, agent):
        agent_idx = self.agents.index(agent)

        def fill_other_info():
            num_players = len(self.agents)
            for i in range(1, num_players):
                next_agent = self.agents[(agent_idx + i) % num_players]
                for j, property_slot in enumerate(self.state[next_agent]["property_slots"]):
                    self.state[agent]["other_properties"][i-1][j] = property_slot
                self.state[agent]["other_banks"][i-1] = self.state[next_agent]["bank"]

        def flattened_observation():
            hand = np.zeros(12)
            bank = np.zeros(50)
            properties = np.zeros(8 * NUM_PROPERTY_SLOTS)
            other_properties = np.zeros(8 * NUM_PROPERTY_SLOTS * (len(self.possible_agents) - 1))
            other_banks = np.zeros(50 * (len(self.agents) - 1))
            for i, card_id in enumerate(self.state[agent]["hand"]):
                hand[i] = card_id
            for i, bank_value in enumerate(self.state[agent]["bank"]):
                bank[i] = bank_value
            for i, property_slot in enumerate(self.state[agent]["property_slots"]):
                for j, card_id in enumerate(property_slot):
                    properties[i*8 + j] = card_id
            for k in range(2):
                for i, property_slot in enumerate(self.state[agent]["other_properties"][k]):
                    for j, card_id in enumerate(property_slot):
                        other_properties[i*8 + j + (k * 4 * NUM_PROPERTY_SLOTS)] = card_id
                for i, bank_value in enumerate(self.state[agent]["other_banks"][k]):
                    other_banks[i] = bank_value
            return np.concatenate((hand, bank, properties, other_properties, other_banks))

        self.fill_action_mask(agent)
        fill_other_info()
        return flattened_observation()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)

        num_players = options["num_players"] if options and "num_players" in options else 3
        self.agents = self.possible_agents[:num_players]
        self.curr_agent_index = 0
        self.current_agent_turns_taken = -1
        self.agent_selection = self.agents[self.curr_agent_index]
        self.next_agent_queue = deque()

        self.completed_properties = {agent: set() for agent in self.agents}

        self.return_to_player = -1
        self.action_type = ""
        self.in_action_cycle = False
        self.action_completed = False
        self.action_info = {}

        self.amount_owed = 0
        self.pay_from_property = False

        self.must_discard = False
        self.deck_ran_out = False

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.state = {agent: {
            "hand": [],
            "property_slots": [[] for _ in range(10)],
            "bank": [],
            "other_properties": [[[] for _ in range(10)] for _ in range(num_players - 1)],
            "other_banks": [[] for _ in range(num_players - 1)],
        } for agent in self.agents}

        for _ in range(3):
            for agent in self.agents:
                player_hand = self.state[agent].get("hand", [])
                player_hand.append(self.draw_from_deck())
                self.state[agent]["hand"] = player_hand

        for _ in range(2):
            for agent in self.agents:
                player_hand = self.state[agent].get("hand", [])
                player_hand.append(self.deck.popleft())

        first_agent = self.agents[0]
        for i in range(2):
            self.state[first_agent]["hand"].append(self.draw_from_deck())

        self.observations = {agent: 0 for agent in self.agents}

    def take_action_on_player(self, action_type, target_player, action_info: dict):
        target_player = self.agents[(self.curr_agent_index + target_player + 1) % len(self.agents)]
        self.return_to_player = self.agent_selection
        self.action_type = action_type
        self.next_agent_queue.append(target_player)
        self.action_info = action_info

    def select_agent(self, agent):
        self.agent_selection = agent
        self.curr_agent_index = self.agents.index(agent)

    def next_turn(self):
        if self.winner >= 0 or self.deck_ran_out:
            next_agent_idx = (self.curr_agent_index + 1) % len(self.agents)
            self.curr_agent_index = next_agent_idx
            self.agent_selection = self.agents[next_agent_idx]
            self.terminations[self.agent_selection] = True
            self._cumulative_rewards[self.agent_selection] = win_reward \
                if self.winner == self.curr_agent_index else loss_reward
            if self.deck_ran_out:
                self._cumulative_rewards[self.agent_selection] = deck_run_out_reward
            return

        if self.in_action_cycle:
            if self.action_completed:
                if self.next_agent_queue:
                    self.action_completed = False
                    self.select_agent(self.next_agent_queue.popleft())
                    return
                else:
                    self.in_action_cycle = False
                    self.action_completed = True
                    self.select_agent(self.return_to_player)
            else:
                return
        elif self.next_agent_queue:
            self.select_agent(self.next_agent_queue.popleft())
            self.in_action_cycle = True
            return

        if not self.must_discard:
            self.current_agent_turns_taken += 1
        if self.current_agent_turns_taken == 2:
            if len(self.state[self.agent_selection]["hand"]) > 7:
                self.must_discard = True
                return
            else:
                self.must_discard = False

            next_agent = (self.curr_agent_index + 1) % len(self.agents)
            self.curr_agent_index = next_agent
            self.agent_selection = self.agents[next_agent]
            self.current_agent_turns_taken = -1
            if not self.state[self.agent_selection]["hand"]:
                for i in range(5):
                    self.state[self.agent_selection]["hand"].append(self.draw_from_deck())
            else:
                for i in range(2):
                    self.state[self.agent_selection]["hand"].append(self.draw_from_deck())

    def step(self, action):
        self._cumulative_rewards[self.agent_selection] = 0
        if action is None:
            self.terminations[self.agent_selection] = True
            self.next_turn()
            return

        assert self.infos[self.agent_selection]["action_mask"][action] == 1

        def get_card_from_hand(card_id):
            try:
                idx = hand.index(card_id)
            except ValueError:
                raise ValueError(f"{card_id} not in hand: {hand}. Action was {action}")
            hand[idx], hand[-1] = hand[-1], hand[idx]
            return hand.pop()

        def accept_action():
            action_type = self.action_type
            action_agent = self.return_to_player
            target_property_slot = self.action_info["target_property_slot"] \
                if "target_property_slot" in self.action_info else None

            if action_type == "deal_breaker":
                own_property_slot = self.state[agent]["property_slots"][target_property_slot]
                action_agent_property_slot = self.state[action_agent]["property_slots"][target_property_slot]
                while len(own_property_slot) > 0:
                    action_agent_property_slot.append(own_property_slot.pop())
                self.action_completed = True
                self._cumulative_rewards[agent] = rare_action_reward

            elif action_type == "sly_deal":
                property_slot = self.action_info["target_property_slot"]
                properties = self.state[self.return_to_player]["property_slots"]

                target_property_slot = self.state[agent]["property_slots"][property_slot]
                target_property_slot.sort()
                card = target_property_slot.pop()
                properties[property_slot].append(card)
                self.action_completed = True

            elif action_type == "forced_deal":
                agent_property_slot, target_property_slot = (self.action_info["agent_property_slot"],
                                                           self.action_info["target_property_slot"])

                properties = self.state[self.return_to_player]["property_slots"]
                target_properties = self.state[agent]["property_slots"]
                agent_properties = properties[agent_property_slot]
                agent_properties.sort(reverse=True)
                target_properties = target_properties[target_property_slot]
                target_properties.sort()

                agent_card = agent_properties.pop()
                target_card = target_properties.pop()

                self.state[self.return_to_player]["property_slots"][target_property_slot].append(target_card)
                self.state[agent]["property_slots"][agent_property_slot].append(agent_card)
                self.action_completed = True

            elif action_type == "debt_collector" or action_type == "birthday" or action_type == "rent":
                amount_owed = self.action_info["amount_owed"]
                if sum(bank) <= amount_owed:
                    cards = bank.copy()
                    self.state[agent]["bank"] = []
                else:
                    cards = pay_from_bank(
                        bank=self.state[agent]["bank"],
                        amount=amount_owed,
                    )

                self.state[self.return_to_player]["bank"] += cards
                amount_owed -= sum(cards)
                if amount_owed <= 0 or not any(property_slot for property_slot in self.state[agent]["property_slots"]):
                    self.action_completed = True
                    self.pay_from_property = False
                else:
                    self.pay_from_property = True
                    self.action_completed = False

        agent = self.agent_selection
        agent_state = self.state[agent]
        hand = agent_state["hand"]
        bank = agent_state["bank"]
        properties = agent_state["property_slots"]

        logger.info(f"Action index: {action}")
        action_idx = action
        action, target_player, property_slot = get_action(action)
        self.action_counts[agent][action] = self.action_counts[agent].get(action, 0) + 1
        logger.info(f"Action taken: {action}, target_player: {target_player}, property_slot: {property_slot}")
        if action == "bank":
            card_id = action_idx
            if card_id >= 62:
                card_id += 39
            card = get_card_from_hand(card_id)
            bank.append(CARD_VALUES[card])
            if card_id > 19:
                self._cumulative_rewards[agent] = bank_reward

        elif action == "deal_breaker": # Check for just say no
            for card_id in hand:
                if card_id in (20, 21):
                    self.discard_pile.append(get_card_from_hand(card_id))
                    break
            self.take_action_on_player(
                "deal_breaker",
                target_player,
                {"target_property_slot": property_slot})

        elif action == "just_say_no":
            self.action_completed = True

            self._cumulative_rewards[agent] = rare_action_reward

        elif action == "pass_go":
            for card_id in hand:
                if 25 <= card_id <= 34:
                    self.discard_pile.append(get_card_from_hand(card_id))
                    break
            for _ in range(2):
                hand.append(self.draw_from_deck())

        elif action == "sly_deal": # Deal with stealing wildcards
            for card_id in hand:
                if card_id in (38, 39, 40):
                    self.discard_pile.append(get_card_from_hand(card_id))
                    break
            self.take_action_on_player(
                action_type="sly_deal",
                target_player=target_player,
                action_info={
                    "target_property_slot": property_slot,
                }
            )

        elif action == "forced_deal":
            for card_id in hand:
                if card_id in (35, 36, 37):
                    self.discard_pile.append(get_card_from_hand(card_id))
                    break
            agent_property_slot, target_property_slot = property_slot
            self.take_action_on_player(
                action_type="forced_deal",
                target_player=target_player,
                action_info={
                    "agent_property_slot": agent_property_slot,
                    "target_property_slot": target_property_slot,
                }
            )

        elif action == "debt_collector": # Implement force_pay
            for card_id in hand:
                if card_id in (41, 42, 43):
                    self.discard_pile.append(get_card_from_hand(card_id))
                    break
            next_agent = self.possible_agents[(self.curr_agent_index + target_player) % len(self.possible_agents)]
            logger.info(f"{self.agent_selection} forces {next_agent} to pay 5 million.")
            self.take_action_on_player(
                action_type="debt_collector",
                target_player=target_player,
                action_info={"amount_owed": 5}
            )

        elif action == "birthday":
            for card_id in hand:
                if card_id in (44, 45, 46):
                    self.discard_pile.append(get_card_from_hand(card_id))
                    break

            for target_player in range(len(self.possible_agents) - 1):
                self.take_action_on_player(
                    action_type="birthday",
                    target_player=target_player,
                    action_info={"amount_owed": 2}
                )

        elif action == "rent":
            def calculate_rent_payment(card_id, double):
                property_slot = (card_id - 49) // 2
                max_len = CARDS_TO_COMPLETE[property_slot]
                amount = RENT_VALUES[property_slot][min(max_len-1, len(properties[property_slot]) - 1)]
                if len(properties[property_slot]) >= max_len:
                    for card_id in properties[property_slot]:
                        if card_id in (101, 102, 103):
                            amount += 3
                        elif card_id > 103:
                            amount += 4
                if double:
                    amount += 2
                return amount

            rent_offset, double = property_slot
            card_id = 49 + rent_offset * 2
            logger.info(f"Rent offset is {rent_offset}")
            if card_id not in hand:
                if card_id + 1 in hand:
                    card_id += 1
                else:
                    for n in range(59, 62):
                        if n in hand:
                            card_id = n
                            break
            self.discard_pile.append(get_card_from_hand(card_id))
            logger.info(f"{self.agent_selection} forces {target_player} to pay rent.")
            self.take_action_on_player(
                action_type="rent",
                target_player=target_player,
                action_info={"amount_owed": calculate_rent_payment(card_id, double)}
            )
            self._cumulative_rewards[agent] = common_action_reward

        elif action == "place_property":
            property_card = -1
            for card_id in hand:
                if card_id in CARDS_IN_PROPERTY_SLOTS[property_slot]:
                    property_card = get_card_from_hand(card_id)
                    break
            logger.info(f"Placing property {property_card} on {property_slot}")
            if property_card == -1:
                raise ValueError("Property card not found")
            properties[property_slot].append(property_card)
            self._cumulative_rewards[agent] = place_property_reward


        elif action == "place_wildcard":
            wildcard = -1
            for card_id in hand:
                if card_id in WILDCARD_TO_PROPERTY_SLOTS and property_slot in WILDCARD_TO_PROPERTY_SLOTS[card_id]:
                    wildcard = get_card_from_hand(card_id)
                    break
            if wildcard == -1:
                raise ValueError("Wildcard not found")
            properties[property_slot].append(wildcard)

        elif action == "flip_wildcard":
            flip_to_slot = property_slot
            card_id = -1
            for i, card_id in enumerate(properties[property_slot]):
                if 90 <= card_id <= 98:
                    flip_to_slot = WILDCARD_TO_PROPERTY_SLOTS[card_id][0]
                    if flip_to_slot == property_slot:
                        flip_to_slot = WILDCARD_TO_PROPERTY_SLOTS[card_id][1]
                    properties[property_slot][i], properties[property_slot][-1] = (properties[property_slot][-1],
                                                                                   properties[property_slot][i])
                    card_id = properties[property_slot].pop()
                    break
            logging.info(f"Flipping from {property_slot} to {flip_to_slot}")
            if card_id == -1:
                raise ValueError("Wildcard to flip not found")
            elif property_slot == flip_to_slot:
                raise ValueError("Flipping to same slot")

            properties[flip_to_slot].append(card_id)

        elif action == "place_wildcard_all":
            card = -1
            for card_id in hand:
                if card_id in (99, 100):
                    card = get_card_from_hand(card_id)
                    break
            logger.info(f"Placing wildcard all ({card}) on property_slot {property_slot}")
            if card == -1:
                raise ValueError("Wildcard not found")
            properties[property_slot].append(card)

        elif action == "move_wildcard_all":
            src, dst = property_slot
            card_id = -1
            for i, card_id in enumerate(properties[src]):
                if card_id in (99, 100):
                    properties[src][i], properties[src][-1] = (properties[src][-1], properties[src][i])
                    card_id = properties[src].pop()
                    break
            logger.info(f"Moving wildcard ({card_id}) from {src} to {dst}")
            if card_id == -1:
                raise ValueError(f"Wildcard_all to move not found in properties: {properties} (src: {src}, dst: {dst})\n"
                                 f"action_mask: {self.infos[agent]['action_mask'][offsets['move_wildcard_all']: offsets['move_wildcard_all'] + 100]}")
            properties[dst].append(card_id)

        elif action == "place_house":
            card = -1
            for card_id in hand:
                if card_id in (101, 102, 103):
                    card = get_card_from_hand(card_id)
                    break
            logger.info(f"Placing house ({card}) on property_slot {property_slot}")
            if card == -1:
                raise ValueError("House not found")
            properties[property_slot].append(card)
            self._cumulative_rewards[agent] = rare_action_reward

        elif action == "place_hotel":
            card = -1
            for card_id in hand:
                if card_id in (104, 105):
                    card = get_card_from_hand(card_id)
                    break
            logger.info(f"Placing hotel ({card}) on property_slot {property_slot}")
            if card == -1:
                raise ValueError("Hotel not found")
            properties[property_slot].append(card)
            self._cumulative_rewards[agent] = rare_action_reward

        elif action == "pay_sum_from_property":
            target_property_slot = self.state[agent]["property_slots"][property_slot]
            target_property_slot.sort(reverse=True)
            property_card = target_property_slot.pop()
            self.state[self.return_to_player]["property_slots"][property_slot].append(property_card)
            value = CARD_VALUES[property_card]
            self.amount_owed -= value
            if self.amount_owed <= 0:
                self.pay_from_property = False
                self.action_completed = True

        elif action == "accept_action":
            accept_action()
            self._cumulative_rewards[agent] = accept_action_reward

        elif action == "discard_card":
            card_id = property_slot
            self.discard_pile.append(get_card_from_hand(card_id))
            self._cumulative_rewards[agent] = discard_reward

        elif action == "do_nothing":
            self.current_agent_turns_taken = 1
            self._cumulative_rewards[agent] = common_action_reward

        else:
            raise ValueError(f"Unrecognized action: {action}")

        for i, property_slot in enumerate(properties):
            if len(property_slot) >= CARDS_TO_COMPLETE[i] and i not in self.completed_properties[agent]:
                self.completed_properties[agent].add(i)
                self._cumulative_rewards[agent] = complete_set_reward
            if len(property_slot) < CARDS_TO_COMPLETE[i] and i in self.completed_properties[agent]:
                self.completed_properties[agent].remove(i)
                # self._cumulative_rewards[agent] = -complete_set reward
        if len(self.completed_properties[agent]) == 3:
            self.winner = self.agents.index(agent)

        self.next_turn()
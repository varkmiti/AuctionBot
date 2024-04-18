import numpy as np
import polars as pl
import tqdm

class AuctionEnvironment:
    def __init__(self, max_bid: float = 1.0):
        self.max_bid = max_bid
    
    def determine_winners(self, bids: np.array) -> (int, float):
        max_bid = np.max(bids)
        max_count = np.sum(bids == max_bid)
        if max_count > 1:
            return -1, 0.0  # Tie, no winner
        else:
            non_max_bids = bids[bids < max_bid]
            second_highest_bid = np.max(non_max_bids) if non_max_bids.size > 0 else 0.0
            winner_index = np.argmax(bids)
            return winner_index, second_highest_bid

def simulate_auction_async(agents, env):
    bids = [agent.choose_action() for agent in agents]
    winner_index, payment = env.determine_winners(np.array(bids))
    winner = agents[winner_index] if winner_index != -1 else None
    for i, agent in enumerate(agents):
        reward = agent.type - payment if agent == winner else 0
        agent.update_q_matrix(bids[i], reward)
    return bids, winner_index, payment

def train_agents_async(number_epochs, agents, env):
    for epoch in range(number_epochs):
        simulate_auction_async(agents, env)

def simulate_auction_synchronous(agents, env, epsilon):
    states = [(agent.last_bid, agent.last_other_bids) for agent in agents]
    actions = [agent.choose_action(state) for agent, state in zip(agents, states)]
    winner_index, payment = env.determine_winners(np.array(actions))
    winner = agents[winner_index] if winner_index != -1 else None
    rewards = [agent.calculate_reward(winner, payment) for agent in agents]
    next_states = [(actions[i], [actions[j] for j in range(len(actions)) if j != i]) for i in range(len(agents))]
    
    for agent, state, next_state in zip(agents, states, next_states):
        all_possible_actions = range(agent.num_actions)
        all_possible_rewards = [agent.calculate_reward(winner, payment if idx == winner_index else 0) for idx in range(agent.num_actions)]
        agent.update_q_matrix(state, all_possible_actions, all_possible_rewards, next_state)

    for i, agent in enumerate(agents):
        agent.last_bid = actions[i]
        agent.last_other_bids = [actions[j] for j in range(len(actions)) if j != i]

    return states, actions, winner, payment, rewards

# Example of ensuring data passed to DataFrame is of correct type
def train_agents_synchronous(number_epochs, number_episodes, agents, env, epsilon):
    results = pl.DataFrame({
        "epoch": [],
        "episode": [],
        "agent": [],
        "type": [],
        "bid": [],
        "winner": [],
        "payment": [],
        "reward": [],
    })

    for epoch in tqdm.tqdm(range(number_epochs), desc="Epochs"):
        for episode in tqdm.tqdm(range(number_episodes), desc="Episodes", leave=False):
            state, actions, winner, payment, rewards = simulate_auction_synchronous(agents, env, epsilon)
            
            # Convert results to proper types and structure
            results = results.vstack(
                pl.DataFrame({
                    "epoch": [epoch] * len(agents),
                    "episode": [episode] * len(agents),
                    "agent": list(range(len(agents))),
                    "type": [int(agent.type) for agent in agents],  # Ensuring type is int
                    "bid": [int(action) for action in actions],  # Ensuring bids are int
                    "winner": [winner] * len(agents),  # Assuming winner index or -1 if no winner
                    "payment": [float(payment)] * len(agents),  # Ensuring payment is float
                    "reward": [float(reward) for reward in rewards],  # Ensuring rewards are float
                })
            )

    return results

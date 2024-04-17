class AuctionEnvironment:
    def __init__(self, max_bid: float = 1.0):
        self.max_bid = max_bid
    
    def determine_winners(self, bids: numpy.ndarray) -> Tuple(int, float):
        if len(np.max(bids)) != 1:
            # Multiple winners, no one wins
            return 0, 0
        # GSP Auction, so the highest bid wins but pays the second highest bid
        else: 
            return np.argmax(bids), np.sort(bids)[-2]

def simulate_auction(agents: list, env, epsilon):
    types = [agent.sample_type() for agent in agents]
    bids = [agent.choose_action() for agent in agents]
    winner, payment = env.determine_winners(bids)
    rewards = [agent.calculate_reward(payment) for agent in agents]
    for i, agent in enumerate(agents):
        agent.update_q_matrix(winner, bids[i], rewards[i], winner)
    return types, bids, winner, payment, rewards

def train_agents(number_epochs, number_episodes, agents, env, epsilon):
    # we will store the results of the simulation in a Polars DataFrame
    results = pl.DataFrame(
        {
            "epoch": pl.Series("epoch", [0]),
            "episode": pl.Series("episode", [0]),
            "agent": pl.Series("agent", [0]),
            "type": pl.Series("type", [0.0]),
            "bid": pl.Series("bid", [0.0]),
            "winner": pl.Series("winner", [0]),
            "payment": pl.Series("payment", [0.0]),
            "reward": pl.Series("reward", [0.0]),
        }
    )

    for epoch in tqdm(range(number_epochs), desc="Epochs"):
        for episode in tqdm(range(number_episodes), desc= "Episodes", leave = False):
            result = simulate_auction(agents, env, epsilon)
            # Put the results into the Results DataFrame
            results = results.vstack(
                pl.DataFrame(
                    {
                        "epoch": pl.Series("epoch", [epoch]),
                        "episode": pl.Series("episode", [episode]),
                        "agent": pl.Series("agent", list(range(len(agents)))),
                        "type": pl.Series("type", result[0]),
                        "bid": pl.Series("bid", result[1]),
                        "winner": pl.Series("winner", result[2]),
                        "payment": pl.Series("payment", result[3]),
                        "reward": pl.Series("reward", result[4]),
                    }
                )
            )
    return results

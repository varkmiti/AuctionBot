import numpy as np
import scipy.stats as stats

class NoInfoAgent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.1, 
                type_distribution: callable = stats.uniform(0, 1), num_actions: int = int(1e3), epsilon: float = 1e-1):
        self.q_matrix = np.zeros(num_actions)  # Simplified Q-matrix with one dimension
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.type_distribution = type_distribution
        self.num_actions = num_actions
        self.type = self.sample_type()
        self.epsilon = epsilon

    def sample_type(self):
        return self.type_distribution.rvs()

    def choose_action(self):
        # Choose action based on a simple epsilon-greedy strategy from the Q-matrix
        if np.random.rand() < self.epsilon:  # Using a fixed epsilon value of 0.1 for exploration
            return np.random.choice(np.linspace(0, 1, self.num_actions))
        else:
            return np.argmax(self.q_matrix)

    def update_q_matrix(self, action, reward):
        current_q_value = self.q_matrix[action]
        self.q_matrix[action] = current_q_value + self.learning_rate * (reward - current_q_value)


class FullInfoAgent:
    def __init__(self, learning_rate, discount_factor, type_distribution, epsilon, num_actions):
        self.q_matrix = np.zeros((1000, num_actions))  # Example: Adjust size if needed
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.type_distribution = type_distribution
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.type = self.sample_type()
        self.last_bid = 0
        self.last_other_bids = []

    def sample_type(self):
        return self.type_distribution.rvs()

    def state_to_index(self, state):
        # Convert state elements to a tuple if they are lists
        state = tuple(state)  # Assuming state is a tuple like (last_bid, last_other_bids)
        state = (state[0], tuple(state[1]))  # Ensure the second element, if a list, is converted to tuple
        # Use a simple hash-based conversion; adjust as needed for your state complexity
        return hash(state) % 1000  # Modulo number of states; this is just an example

    def choose_action(self, state):
        state_index = self.state_to_index(state)  # Convert state to a valid index
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_actions)  # Exploratory random action
        else:
            return np.argmax(self.q_matrix[state_index, :])  # Exploitative action based on Q-matrix

    def calculate_reward(self, winner, payment):
        if self == winner:
            return self.type - payment
        return 0

    def update_q_matrix(self, state, all_actions, all_rewards, next_state):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        for action, reward in zip(all_actions, all_rewards):
            current_q_value = self.q_matrix[state_index, action]
            future_max_q = np.max(self.q_matrix[next_state_index, :])
            self.q_matrix[state_index, action] = (1 - self.learning_rate) * current_q_value + \
                                           self.learning_rate * (reward + self.discount_factor * future_max_q)

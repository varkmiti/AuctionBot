import numpy as np
import polars as pl
import scipy.stats as stats


class FullInfoAgent:
    def __init__(self, 
        learning_rate, 
        discount_factor, 
        type_distribution: callable = stats.uniform(0, 1),
        epsilon: float = 1e-1):
            self.q_matrix = np.zeros((1e3, 1e3))
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.type_distribution = type_distribution
            self.type = self.sample_type()
            self.epsilon = epsilon
        
    def sample_type(self):
        return self.type_distribution.rvs()

    def calculate_reward(self, result):
        return self.type - result

    # Since the agent has full information, we can update the Q matrix synchrously and statefully
    def update_q_matrix(self, state, action, result, next_state):
        reward = self.calculate_reward(result)
        self.q_matrix[state, action] = self.q_matrix[state, action] + self.learning_rate * (reward + self.discount_factor * np.max(self.q_matrix[next_state, :]) - self.q_matrix[state, action])
        return self.q_matrix
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.uniform(0, self.type)
        else:
            return np.argmax(self.q_matrix[state, :])


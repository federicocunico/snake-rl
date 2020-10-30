from typing import Type
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from src.snake_structures import GameState, Direction

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Deep Learning paramteres
# -------------------------------------------------
lr = 5e-4  # 0.0005
epochs = 150

# Reinforcement Learning Parameters
# -------------------------------------------------
memory_size = 2500
batch_size = 1000
epsilon_decay_linear = 1 / 75


class SnakeAgent:
    def __init__(self):
        # Neural Network Model
        self.model = None
        self.optim = None
        self.loss = None
        self.network()

        self.reward = 0
        self.gamma = 0.9
        self.short_memory = np.array([])
        self.epsilon = 1
        self.is_init = False

    def network(self):
        input_layer = nn.Linear(in_features=11, out_features=50)
        hidden_layer_1 = nn.Linear(in_features=50, out_features=300)
        hidden_layer_2 = nn.Linear(in_features=300, out_features=50)
        output_layer = nn.Linear(in_features=50, out_features=3)

        model = nn.Sequential(
            input_layer,
            nn.ReLU(),
            hidden_layer_1,
            nn.ReLU(),
            hidden_layer_2,
            nn.ReLU(),
            output_layer,
            nn.Softmax()
        )
        # Place on GPU before init of optim otherwise runtime error
        model = model.to(device)

        optim = Adam(model.parameters(), lr=0.0005)
        loss = nn.MSELoss()

        self.model = model
        self.optim = optim
        self.loss = loss

    def predict(self, input_features):
        raise NotImplementedError

    @staticmethod
    def get_state(game_state: Type[GameState]):
        curr_pos = game_state.snake_head_position

        # Check if there is a proximity danger
        up_curr_pos = curr_pos + Direction.UP
        right_curr_pos = curr_pos + Direction.RIGHT
        left_curr_pos = curr_pos + Direction.LEFT
        danger_straight = game_state.is_collided(up_curr_pos)
        danger_right = game_state.is_collided(right_curr_pos)
        danger_left = game_state.is_collided(left_curr_pos)

        # Which direction is moving
        is_moving_up = game_state.candidate_snake_head_direction == Direction.UP
        is_moving_right = game_state.candidate_snake_head_direction == Direction.RIGHT
        is_moving_down = game_state.candidate_snake_head_direction == Direction.DOWN
        is_moving_left = game_state.candidate_snake_head_direction == Direction.LEFT

        # Food closest
        is_food_up, is_food_right, is_food_down, is_food_left = game_state.get_closest_food_direction()

        res = [
            danger_straight,
            danger_right,
            danger_left,

            is_moving_up,
            is_moving_right,
            is_moving_down,
            is_moving_left,

            is_food_up,
            is_food_right,
            is_food_down,
            is_food_left
        ]
        return res

    def set_reward(self, eaten, crashed):
        self.reward = 0
        if crashed:
            self.reward = -10
            return self.reward
        if eaten:
            self.reward = 10
            return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)

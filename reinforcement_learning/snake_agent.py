import collections
import random
from typing import Type
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from reinforcement_learning.networks import linear_q1, linear_q2
from reinforcement_learning.utils import to_categorical
from src.snake_structures import GameState, Direction

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Deep Learning paramteres
# -------------------------------------------------
lr = 5e-4  # 0.0005
epochs = float('inf')  # 150

# Reinforcement Learning Parameters
# -------------------------------------------------
memory_size = 2500
batch_size = 1000
epsilon_decay_linear = 1 / 75


class SnakeAgent:
    def __init__(self):
        # Neural Network Model
        self.model = None
        self.optimizer = None
        self.loss = None
        self.network()

        self.reward = 0
        self.gamma = 0.9
        self.short_memory = np.array([])
        self.memory = collections.deque(maxlen=memory_size)
        self.epsilon = 0
        self.is_init = False

    def network(self):
        print('Initializing network')
        # model = linear_q1()
        model = linear_q2()

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        model.apply(init_weights)

        print('Network created')
        # Place on GPU before init of optim otherwise runtime error
        model = model.to(device)
        print(f'Network placed on {device}')

        optim = Adam(model.parameters(), lr=0.0005)
        loss = nn.MSELoss()

        self.model = model
        self.optimizer = optim
        self.loss = loss

    def predict(self, input_features):

        if isinstance(input_features, np.ndarray):
            input_features = torch.from_numpy(input_features)
        input_features = input_features.to(device)

        output = self.model(input_features)
        # print(f"Network output: {output}")
        return output

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
        res = torch.FloatTensor(res).reshape((1, 11))
        return res

    def set_reward(self, eaten, crashed):
        self.reward = 0
        if crashed:
            self.reward = -10
            return self.reward
        if eaten:
            self.reward = 10
            return self.reward
        return self.reward

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self):
        memory = self.memory
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.train_short_memory(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        # state = torch.tensor(state, dtype=torch.float)
        # next_state = torch.tensor(next_state, dtype=torch.float)
        # action = torch.tensor(action, dtype=torch.long)
        # reward = torch.tensor(reward, dtype=torch.float)
        target = reward

        next_state = next_state.to(device)
        state = state.to(device)

        if not done:
            target = reward + self.gamma * torch.max(self.model(next_state))
        pred = self.model(state)
        target_f = pred.clone()
        target_f[0][torch.argmax(action).item()] = target
        loss = self.loss(target_f, pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # # if not done:
        # target = reward + self.gamma * torch.amax(self.predict(next_state)[0])
        # # else:
        # #     # target = torch.zeros((1, 3), requires_grad=True).to(device)
        # #     target = self.predict(next_state)
        # #     # target_clone = target.clone()
        # #     # target[0][torch.argmax(action)] = reward
        # #     # target = target_clone
        #
        # target_f = self.predict(state)
        #
        # # target_clone = target_f.clone()
        # # target_clone[0][torch.argmax(action)] = target
        #
        # # s = action.reshape((1, -1)).to(device)
        # # t = target_clone.reshape((1, -1))
        # s = target_f
        # t = target
        #
        # loss = self.loss(t, s)
        #
        # print(loss.detach().cpu().numpy())
        #
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

    @staticmethod
    def to_categorical(prediction, num_classes):
        pred = torch.argmax(prediction[0])
        return to_categorical(pred, num_classes)

    def get_next_move(self, last_state):
        # perform random actions based on agent.epsilon, or choose the action
        if random.uniform(0, 1) < self.epsilon:
            final_move_categorical = to_categorical(random.randint(0, 2), num_classes=3)
        else:
            # predict action based on the old state
            prediction = self.predict(last_state)
            final_move_categorical = self.to_categorical(prediction, num_classes=3)
        # order of final move is....? Se devo scegliere io: STRAIGHT, RIGHT, LEFT
        # TODO: final_move is [1,0,0] or [0,1,0] or [0,0,1]. Convert to Direction
        final_move = final_move_categorical.detach().cpu().numpy()
        final_move = self.categorical_to_direction(final_move)

        assert final_move in [Direction.UP, Direction.LEFT, Direction.RIGHT, Direction.DOWN]
        # print(f'Next Move: {Direction.name_of(final_move)}')
        return final_move, final_move_categorical

    @staticmethod
    def categorical_to_direction(final_move):
        # Arbitrary
        # [1,0,0] => straight w.r.t. current direction
        # [0,1,0] => right w.r.t. current direction
        # [0,0,1] => left w.r.t. current direction
        def get_relative_direction():
            # todo: use max along axis of (1x3) numpy array
            if final_move[0] == 1 and final_move[1] == 0 and final_move[2] == 0:
                return Direction.STRAIGHT
            elif final_move[0] == 0 and final_move[1] == 1 and final_move[2] == 0:
                return Direction.RIGHT
            elif final_move[0] == 0 and final_move[1] == 0 and final_move[2] == 1:
                return Direction.LEFT

        curr_dir = GameState.snake_head_direction

        rel_dir = get_relative_direction()
        if rel_dir == Direction.STRAIGHT:
            return curr_dir

        if rel_dir == Direction.RIGHT:
            if curr_dir == Direction.UP:
                return rel_dir
            elif curr_dir == Direction.RIGHT:
                return Direction.DOWN
            elif curr_dir == Direction.DOWN:
                return Direction.LEFT
            elif curr_dir == Direction.LEFT:
                return Direction.UP

        if rel_dir == Direction.LEFT:
            if curr_dir == Direction.UP:
                return rel_dir
            elif curr_dir == Direction.RIGHT:
                return Direction.UP
            elif curr_dir == Direction.DOWN:
                return Direction.RIGHT
            elif curr_dir == Direction.LEFT:
                return Direction.DOWN

        raise ValueError

    def direction_to_categorical(self, action):
        # Arbitrary
        # [1,0,0] => straight w.r.t. current direction
        # [0,1,0] => right w.r.t. current direction
        # [0,0,1] => left w.r.t. current direction

        if action == Direction.UP:
            return

        return

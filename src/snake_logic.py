import time
import random
import numpy as np
from random import randint
from threading import Thread

from reinforcement_learning.snake_agent import SnakeAgent
from reinforcement_learning.utils import to_categorical
from src.snake_structures import GameState as g, Direction
from src.snake_structures import GridPosition

adjust_speed = True


class GameLogic(Thread):
    def __init__(self, train_agent=False):
        super().__init__()
        self.train_agent = train_agent
        self.agent = SnakeAgent()

        self.scores = []

    min_interval = 0.03
    max_interval = 0.2

    def run(self):
        while True:

            if self.train_agent:
                if not self.agent.is_init:
                    state_init1 = self.agent.get_state(g)
                    eaten, collided = self.update(state_init1)
                    state_init2 = self.agent.get_state(g)
                    reward1 = self.agent.set_reward(eaten, collided)
                    # self.agent.remember(state_init1, action, reward1, state_init2, game.crash)
                    # self.agent.replay_new(agent.memory, batch_size)
                    self.agent.is_init = True
                    continue

                old_state = self.agent.get_state(g)
                eaten, collided = self.update(old_state)
                new_state = self.agent.get_state(g)
                reward = self.agent.set_reward(eaten, collided)
                # self.agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)
                # self.agent.remember(state_old, final_move, reward, state_new, game.crash)

                if collided:
                    # self.agent.replay_new(agent.memory, batch_size)
                    pass
            else:
                eaten, collided = self.update()

            if not self.train_agent:
                # Adjust difficulty
                if adjust_speed:
                    interval = self.max_interval - g.difficulty_level * 0.01
                    if interval < self.min_interval:
                        interval = self.min_interval
                else:
                    interval = self.max_interval
            else:
                interval = 0

            # update action for training
            if self.train_agent:
                pass

            time.sleep(interval)

    def update(self, last_state=None):
        # is train
        train = self.train_agent
        if train:
            # perform random actions based on agent.epsilon, or choose the action
            if random.uniform(0, 1) < self.agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                # predict action based on the old state
                assert last_state is not None
                prediction = self.agent.predict(last_state.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)
            # order of final move is....? Se devo scegliere io: STRAIGHT, RIGHT, LEFT
            # TODO: final_move is [1,0,0] or [0,1,0] or [0,0,1]. Convert to Direction
            g.candidate_snake_head_direction = final_move

        apple_eaten = False

        g.snake_head_direction = g.candidate_snake_head_direction
        g.snake_head_position = g.snake_head_position + g.snake_head_direction

        # if g.snake_head_position.x < 0 \
        #         or g.snake_head_position.y < 0 \
        #         or g.snake_head_position.x >= g.game_width \
        #         or g.snake_head_position.y >= g.game_height:
        #     is_collided = True
        #
        # if g.snake_head_position in g.snake_body:
        #     is_collided = True
        is_collided = g.is_collided(g.snake_head_position)

        g.snake_body.insert(0, g.snake_head_position.copy())

        # Food eaten
        if g.snake_head_position in g.apples_positions:
            apple_eaten = True
            g.score += 1
            g.difficulty_level += 1
            g.apples_positions.remove(g.snake_head_position)

        if len(g.snake_body) > g.min_snake_body_length and not apple_eaten:
            g.snake_body.pop(-1)

        if is_collided:
            self.reset()

        if len(g.apples_positions) < 1:
            g.apples_positions.append(GridPosition.random_empty_pos())
        return apple_eaten, is_collided

    def reset(self):
        # reset snake position
        g.snake_head_position = GridPosition(g.snake_start_pos_x, g.snake_start_pos_y)
        g.snake_body = [GridPosition(g.snake_start_pos_x, g.snake_start_pos_y)]  # new, not reference

        # reset direction, difficulty and score
        g.candidate_snake_head_direction = Direction.LEFT
        g.difficulty_level = 1
        if self.train_agent:
            self.scores.append(g.score)
        g.score = 0

        # reset food positions
        g.apples_positions.clear()
        g.apples_positions.append(GridPosition.random_empty_pos())



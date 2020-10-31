import time
import random
import numpy as np
from random import randint
from threading import Thread

from reinforcement_learning.snake_agent import SnakeAgent, epsilon_decay_linear, epochs
from src.snake_structures import GameState as g, Direction
from src.snake_structures import GridPosition

adjust_speed = False


def direction_sum(dir1, dir2):
    return [dir1[0] + dir2[0], dir1[1] + dir2[1]]


class GameLogic(Thread):
    def __init__(self, train_agent=False):
        super().__init__()
        self.train_agent = train_agent
        self.agent = SnakeAgent()

        self.scores = []
        self.counter = 0
        self.has_collided = False
        self._stop = False

        self.reset_food_on_reset = True

    min_interval = 0.03
    max_interval = 0.2

    def stop(self):
        self._stop = True

    def init_game(self):
        self.reset(is_init=True)

        state_init1 = self.agent.get_state(g)
        eaten, collided = self.update(state_init1)
        action = g.last_move_categorical
        state_init2 = self.agent.get_state(g)
        reward1 = self.agent.set_reward(eaten, collided)

        self.agent.remember(state_init1, action, reward1, state_init2, collided)
        self.agent.replay_new()

    def run(self):
        self.init_game()

        while not self._stop:

            if self.train_agent:
                if self.has_collided:
                    self.has_collided = False
                    if self.counter >= epochs:
                        break
                    else:
                        self.counter += 1
                        self.agent.epsilon = 1 - (self.counter * epsilon_decay_linear)

                old_state = self.agent.get_state(g)
                eaten, collided = self.update(old_state)
                action = g.last_move_categorical
                new_state = self.agent.get_state(g)
                reward = self.agent.set_reward(eaten, collided)
                self.agent.train_short_memory(old_state, action, reward, new_state, collided)
                self.agent.remember(old_state, action, reward, new_state, collided)

                if collided:
                    self.agent.replay_new()

            else:
                # not training
                self.update()

            if not self.train_agent:
                # Adjust difficulty
                if adjust_speed:
                    interval = self.max_interval - g.difficulty_level * 0.01
                    if interval < self.min_interval:
                        interval = self.min_interval
                else:
                    interval = self.max_interval
            else:
                # Train speed
                interval = 0
                # interval = self.max_interval

            time.sleep(interval)

        print('Finished')

    def update(self, last_state=None):
        # is train
        train = self.train_agent
        if train:
            assert last_state is not None
            next_move, final_move_categorical = self.agent.get_next_move(last_state)
            g.candidate_snake_head_direction = next_move
            g.last_move_categorical = final_move_categorical

        apple_eaten = False

        if direction_sum(g.candidate_snake_head_direction, g.snake_head_direction) == [0, 0]:
            g.candidate_snake_head_direction = g.snake_head_direction
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
            print('Dead!')
            print('=' * 25)
            self.reset()

        if len(g.apples_positions) < 1:
            g.apples_positions.append(GridPosition.random_empty_pos())
        return apple_eaten, is_collided

    def reset(self, is_init=False):
        if not is_init:
            self.has_collided = True

        # reset snake position
        g.snake_head_position = GridPosition(g.snake_start_pos_x, g.snake_start_pos_y)
        g.snake_body = [GridPosition(g.snake_start_pos_x, g.snake_start_pos_y)]  # new, not reference

        # reset direction, difficulty and score
        g.candidate_snake_head_direction = Direction.RIGHT
        g.difficulty_level = 1
        if self.train_agent:
            self.scores.append(g.score)
        g.score = 0

        # reset food positions
        if self.reset_food_on_reset:
            g.apples_positions.clear()
            # g.apples_positions.append(GridPosition.random_empty_pos())
            g.apples_positions.append(GridPosition(g.food_start_pos_x, g.food_start_pos_y))

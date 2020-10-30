import random
from typing import List


class GridPosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, list) or isinstance(other, GridPosition):
            self.x += other[0]
            self.y += other[1]
        return self

    def __getitem__(self, item):
        if item == 0:
            return self.x
        if item == 1:
            return self.y
        raise IndexError

    def __eq__(self, other):
        if isinstance(other, GridPosition):
            return self.x == other.x and self.y == other.y
        return False

    def copy(self):
        return GridPosition(self.x, self.y)

    @staticmethod
    def random_empty_pos():
        def _temp_new_pos():
            x = random.randint(0, GameState.game_width - 1)
            y = random.randint(0, GameState.game_height - 1)
            return GridPosition(x, y)

        temp_pos = _temp_new_pos()
        while temp_pos in GameState.snake_body:
            temp_pos = _temp_new_pos()

        return temp_pos


class GameState:
    """
    snake_head_direction è codificata come una lista di direzioni da sommare alle posizioni.
    [1, 0] => RIGHT
    [0,-1] => UP
    [0, 1] => DOWN
    [-1, 0] => LEFT

    """
    game_width = 30
    game_height = 30

    min_snake_body_length = 5

    expected_number_of_apples = 1
    apples_positions: List[GridPosition] = []

    snake_start_pos_x = game_width // 2 - 1
    snake_start_pos_y = game_height // 2 - 1

    snake_head_position: GridPosition = GridPosition(snake_start_pos_x, snake_start_pos_y)
    candidate_snake_head_direction = [1, 0]
    snake_head_direction = [1, 0]
    snake_body: List[GridPosition] = [GridPosition(snake_start_pos_x, snake_start_pos_y)]

    score = 0

    difficulty_level = 1

    @staticmethod
    def is_collided(position: GridPosition):
        is_collided = False
        if position.x < 0 \
                or position.y < 0 \
                or position.x >= GameState.game_width \
                or position.y >= GameState.game_height:
            is_collided = True

        if position in GameState.snake_body:
            is_collided = True
        return is_collided

    @staticmethod
    def get_closest_food_direction():
        n_foods = len(g.apples_positions)
        if n_foods > 1:
            import warnings
            warnings.warn('Multiple food found during search for closest food direction. Using the first food only')
        elif n_foods < 1:
            raise RuntimeError('Hey! There is no food! This should not ever happen')

        """
        tables of direction
        x+, y=     right
        x-, y=     left
        x=, y+     up
        x=, y-     down
        x+, y+     up-right
        x+, y-     down-right
        x-, y+     up-left
        x-, y-     down-left
        """
        curr_pos = g.snake_head_position
        food = g.apples_positions[0]

        is_food_left = food.x < curr_pos.x
        is_food_right = food.x > curr_pos.x
        is_food_up = food.y < curr_pos.y
        is_food_down = food.y > curr_pos.y

        return is_food_up, is_food_right, is_food_down, is_food_left


class Direction:
    UP = [0, -1]
    RIGHT = [1, 0]
    DOWN = [0, 1]
    LEFT = [-1, 0]

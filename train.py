# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pygame
import sys
import argparse

from src.snake_structures import GameState, Direction
from reinforcement_learning.snake_agent import epochs
from src.snake_logic import GameLogic, direction_sum


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=bool)
    args = parser.parse_args()

    return not args.test


is_train = get_args()

pygame.init()
pygame.font.init()
font_size = 30
# text_font = pygame.font.SysFont('Comic Sans MS', font_size)
text_font = pygame.font.SysFont('Raleway', font_size)

size = width, height = 800, 800
speed = [0.25, 0.25]
black = 50, 50, 50
gray = 125, 125, 125
white = 255, 255, 255
apple_color = 247, 76, 76
snake_color = 149, 192, 181
border_color = 55, 136, 216

screen = pygame.display.set_mode(size)

clock = pygame.time.Clock()

base_offset = 15
grid_rect_size = 20

text_rendering_row_counter = 0
texts = []

game_logic = GameLogic(train_agent=is_train)
game_logic.start()


def start_game_loop():
    while True:
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.KEYDOWN:
                temp_next_direction = None
                if event.key == pygame.K_UP:
                    # temp_next_direction = [0, -1]
                    temp_next_direction = Direction.UP
                if event.key == pygame.K_DOWN:
                    # temp_next_direction = [0, 1]
                    temp_next_direction = Direction.DOWN
                if event.key == pygame.K_RIGHT:
                    # temp_next_direction = [1, 0]
                    temp_next_direction = Direction.RIGHT
                if event.key == pygame.K_LEFT:
                    # temp_next_direction = [-1, 0]
                    temp_next_direction = Direction.LEFT
                if temp_next_direction is not None \
                        and direction_sum(temp_next_direction, GameState.snake_head_direction) != [0, 0]:
                    GameState.candidate_snake_head_direction = temp_next_direction
                if event.key == pygame.K_q:
                    # pygame.display.quit()
                    game_logic.stop()
                    pygame.quit()
                    sys.exit()

        # rendering
        screen.fill(black)
        print_grid()
        render_game()

        add_text('===========')
        add_text('Snake AI')
        add_text('---------------')
        add_text(f'Score: {GameState.score}')
        add_text(f'Level:  {GameState.difficulty_level}')
        if is_train:
            add_text('---------------')
            add_text(f'Round:  {game_logic.counter}/{epochs}')
            add_text(f'Top Score:  {max(game_logic.scores) if len(game_logic.scores) > 0 else 0}')
            add_text('---------------')
            add_text("")
            if game_logic.counter >= epochs:
                txt = f'Done Training!'
            else:
                txt = "Training..."
            add_text(txt)
        add_text('===========')
        render_text()

        pygame.display.flip()
        clock.tick(60)


def render_text():
    for i, s in enumerate(texts):
        screen.blit(s, ((GameState.game_width * grid_rect_size) + base_offset + 10, i * font_size))
    texts.clear()


def add_text(txt):
    textsurface = text_font.render(txt, False, (173, 12, 51))
    texts.append(textsurface)


def render_game():
    # borders
    pygame.draw.rect(
        screen,
        border_color,
        [
            base_offset,
            base_offset,
            GameState.game_width * grid_rect_size,
            GameState.game_height * grid_rect_size
        ],
        3
    )

    # snake
    for b in GameState.snake_body:
        pygame.draw.rect(
            screen,
            snake_color,
            [
                base_offset + b.x * grid_rect_size,
                base_offset + b.y * grid_rect_size,
                grid_rect_size,
                grid_rect_size
            ]
        )

    # apples
    for a in GameState.apples_positions:
        pygame.draw.rect(
            screen,
            apple_color,
            [
                base_offset + a.x * grid_rect_size,
                base_offset + a.y * grid_rect_size,
                grid_rect_size,
                grid_rect_size
            ]
        )
    pass


def print_grid():
    for i in range(GameState.game_width):
        for j in range(GameState.game_height):
            r = pygame.draw.rect(
                screen,
                gray,
                [
                    (i * grid_rect_size) + base_offset,
                    (j * grid_rect_size) + base_offset,
                    grid_rect_size,
                    grid_rect_size
                ],
                1
            )
            # print(r)


if __name__ == "__main__":
    start_game_loop()

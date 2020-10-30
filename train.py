# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pygame
import sys

from src.snake_structures import GameState
from reinforcement_learning.snake_agent import epochs
from src.snake_logic import GameLogic

pygame.init()
pygame.font.init()
font_size = 30
text_font = pygame.font.SysFont('Comic Sans MS', font_size)

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

game_logic = GameLogic(train_agent=True).start()


def start_game_loop():
    counter = 0

    while counter <= epochs:

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.KEYDOWN:
                temp_next_direction = None
                if event.key == pygame.K_UP:
                    temp_next_direction = [0, -1]
                if event.key == pygame.K_DOWN:
                    temp_next_direction = [0, 1]
                if event.key == pygame.K_RIGHT:
                    temp_next_direction = [1, 0]
                if event.key == pygame.K_LEFT:
                    temp_next_direction = [-1, 0]
                if temp_next_direction is not None and direction_sum(temp_next_direction,
                                                                     GameState.snake_head_direction) != [0, 0]:
                    GameState.candidate_snake_head_direction = temp_next_direction
                if event.key == pygame.K_q:
                    # pygame.display.quit()
                    pygame.quit()
                    sys.exit()

        # rendering
        screen.fill(black)
        print_grid()
        render_game()

        textsurface = text_font.render(f'Score: {GameState.score}', False, (173, 12, 51))
        screen.blit(textsurface, ((GameState.game_width * grid_rect_size) + base_offset + 10, 0))
        textsurface = text_font.render(f'Level:  {GameState.difficulty_level}', False, (173, 12, 51))
        screen.blit(textsurface, ((GameState.game_width * grid_rect_size) + base_offset + 10, font_size))

        pygame.display.flip()
        clock.tick(60)


def direction_sum(dir1, dir2):
    return [dir1[0] + dir2[0], dir1[1] + dir2[1]]


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

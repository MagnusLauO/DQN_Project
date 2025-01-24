# Play BallsGame
import pygame
from BallsGrid import BallsGame

# Initialize game
env = BallsGame()
env.reset()
action = 'none'
exit_program = False

# Game loop
while not exit_program:
    # Render game
    env.render()

    # Process game events
    action = 'none'
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True
            if event.key == pygame.K_RIGHT:
                action = 'right'
            if event.key == pygame.K_LEFT:
                action = 'left'
            if event.key == pygame.K_r:
                env.reset()

    
    # Step the environment
    env.step(action)

# Close the game
env.close()
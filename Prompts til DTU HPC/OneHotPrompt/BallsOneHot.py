import pygame as pg
import numpy as np

class BallsGame:

    # Actions name #
    actions = ['left', 'right', 'none']
    
    ## Init ##
    def __init__(self, cols=5, rows=10):
        pg.init()
        self.cols = cols
        self.rows = rows
        self.reset()
        self.rendering = False
    
    ## Reset fucntion ##
    def reset(self):
        self.score = 0
        self.lives = 3 
        self.player_x = self.cols // 2
        self.balls = []
        self.spawn_timer = 0
        self.base_speed = 0.1
        self.done = False
        self.game_over = False  # Added game over state
        return self.get_state()
    
    ## Get state ##
    def get_state(self):
        # ball exists
        if len(self.balls) == 0:
            ball_x = self.cols // 2
            ball_exists = 0
        else:
            closest_ball = max(self.balls, key=lambda ball: ball.y)
            ball_x = int(closest_ball.x)
            ball_exists = 1
        return (self.player_x, ball_x, ball_exists)
    
    ## Step function ##
    def step(self, action):
        if self.game_over:  # Don't process steps if game is over
            return self.get_state(), 0, True, self.score
            
        # reward = -1 # Reward for each step, Not necessary in our game
        reward = 0
        # Player movement #
        if action == 'left' and self.player_x > 0:
            self.player_x -= 1
            reward -= 2 # penalty for moving
        elif action == 'right' and self.player_x < self.cols - 1:
            self.player_x += 1
            reward -= 2 # penalty for moving
        
        # Spawn balls #
        self.spawn_timer += 1
        if self.spawn_timer >= 40:  # Increased from 5 to 20 for less frequent spawns
            self.spawn_timer = 0
            self.balls.append(Ball(x=np.random.randint(0, self.cols), y=0))
        
        # Ball movement #
        speed = self.base_speed
        for ball in self.balls[:]:
            ball.move(speed, self.score * 0.01)
            
            # Ball pickup and miss #

            if ball.position() == (self.player_x, self.rows - 1):
                self.score += 1
                self.balls.remove(ball)
                reward += 50 # Reward for catching the ball
                
            elif ball.y > self.rows - 1:
                self.lives -= 1
                self.balls.remove(ball)
                reward -= 30 # Penalty for missing the ball
                if self.lives <= 0:
                    self.done = True
                    self.game_over = True  # Game over
                    
        return self.get_state(), reward, self.done, self.score
    

    ## Render function ##
    def render(self):
        if not self.rendering:
            self.init_render()
        
        # Make the screen
        self.screen.fill((255, 255, 255))
        
        # Draw grid
        for i in range(self.rows):
            for j in range(self.cols):
                pg.draw.rect(self.screen, (0, 0, 0), 
                           (j * self.cell_size, i * self.cell_size, 
                            self.cell_size, self.cell_size), 1)
        
        # Draw player
        pg.draw.rect(self.screen, (0, 0, 255),
                    (self.player_x * self.cell_size, 
                     (self.rows-1) * self.cell_size,
                     self.cell_size, self.cell_size))
        
        # Draw balls
        for ball in self.balls:
            pg.draw.circle(self.screen, (255, 0, 0),
                         (int(ball.x * self.cell_size + self.cell_size / 2),
                          int(ball.y * self.cell_size)),
                         10)
        
        # Write score and lives
        font = pg.font.SysFont('Times New Roman', 24)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        lives_text = font.render(f"Lives: {self.lives}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 40))
        

        # Draw game over screen #

        # Help from ChatGPT to learn about overlay.
        if self.game_over:
            overlay = pg.Surface((self.cols * self.cell_size, self.rows * self.cell_size))
            overlay.set_alpha(128)
            overlay.fill((0, 0, 0))
            self.screen.blit(overlay, (0, 0))
            
            # Create text
            game_over_font = pg.font.SysFont('Times New Roman', 48)
            game_over_text = game_over_font.render("GAME OVER", True, (255, 0, 0))
            restart_text = font.render("Press R to restart", True, (255, 255, 255)) # Restart instructions
            
            # Text boxes
            game_over_rect = game_over_text.get_rect(center=(self.cols * self.cell_size // 2, 
                                                            self.rows * self.cell_size // 2))
            restart_rect = restart_text.get_rect(center=(self.cols * self.cell_size // 2, 
                                                       self.rows * self.cell_size // 2 + 50))
            
            # Draw text
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(restart_text, restart_rect)
        
        # udate screen
        pg.display.flip()
        self.clock.tick(30) # 30 FPS
        
    def init_render(self):
        self.cell_size = 60
        self.screen = pg.display.set_mode((self.cols * self.cell_size, 
                                         self.rows * self.cell_size))
        self.clock = pg.time.Clock()
        self.rendering = True
        
    def close(self):
        pg.quit()

## Ball class ##
class Ball:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    # move
    def move(self, speed: float, dt: float):
        self.y += speed + dt

    # get position
    def position(self):
        return int(self.x), int(self.y)
# Space Invaders
import numpy as np
import pygame

class SpaceInvaders():
    # Rendering?
    rendering = False

    # Actions
    actions = ['left', 'right', 'shoot', 'none']

    # Colors
    goodColor = (30, 192, 30)
    badColor = (192, 30, 30)
    gridColor = (192, 192, 192)
    shootColor = (255, 255, 255)

    # Frames per second
    fps = 5
    
    def __init__(self):
        pygame.init()
        self.reset()
                
    def render(self):
        if not self.rendering:
            self.init_render()

        # Limit fps
        self.clock.tick(self.fps)
                 
        # Clear the screen
        self.screen.fill((187,173,160))

        # Draw board
        border = 1
        pygame.draw.rect(self.screen, (187,173,160), pygame.Rect(100,0,600,600))
        w, h = 60-2*border, 60-2*border
        xa, xb = 60, 100+border
        ya, yb = -60, 600-60
        for i in range(10):
            for j in range(10):
                col = self.gridColor
                pygame.draw.rect(self.screen, col, pygame.Rect(xa*i+xb, ya*j+yb, w, h))
                if i==self.player_x and j==0:
                    pygame.draw.rect(self.screen, self.goodColor, pygame.Rect(xa*i+xb, ya*j+yb, w, h))
                if i==self.enemy_x and j==self.enemy_y:
                    pygame.draw.rect(self.screen, self.badColor, pygame.Rect(xa*i+xb, ya*j+yb, w, h))
                if i==self.bullet_x and j==self.bullet_y and self.bullet_on == True:
                    pygame.draw.rect(self.screen, self.shootColor, pygame.Rect(xa*i+xb+w/2-5, ya*j+yb+w/2-5, 10, 10))
                    
        # Draw score
        text = self.scorefont.render("{:}".format(self.score), True, (0,0,0))
        self.screen.blit(text, (790-text.get_width(), 10))        
        
        # Draw game over or you won       
        if self.done:
            if self.won:
                msg = 'Congratulations!'
                col = self.goodColor
            else:
                msg = 'Game over!'
                col = self.badColor
            text = self.bigfont.render(msg, True, col)
            textpos = text.get_rect(centerx=self.background.get_width()/2)
            textpos.top = 300
            self.screen.blit(text, textpos)

        # Display
        pygame.display.flip()
        
    def reset(self):
        self.player_x = np.random.choice(range(10))
        
        self.enemy_x = np.random.choice(range(10))
        self.enemy_y = 9
        self.enemy_direction = np.random.choice([-1, 1])
        
        self.bullet_x = 0
        self.bullet_y = 0
        self.bullet_on = False
        
        self.done = False
        self.won = False
        
        self.tick = 0        
        self.score = 0

        return self.get_state()
    
    def get_state(self):
        return (self.player_x, self.enemy_x, self.enemy_y, self.enemy_direction, self.bullet_x, self.bullet_y, self.bullet_on)

    def close(self):
        # Quit pygame
        pygame.quit()
                 
    def init_render(self):
        # Display mode
        self.screen = pygame.display.set_mode([800, 600])
        # Caption
        pygame.display.set_caption('Space Invaders')
        # Background
        self.background = pygame.Surface(self.screen.get_size())
        # Clock
        self.clock = pygame.time.Clock()
        # Fonts
        self.bigfont = pygame.font.Font(None, 80)
        self.scorefont = pygame.font.Font(None, 30)
        # Rendering on flag
        self.rendering = True
        
    def step(self, action):    
        # Update the clock tick    
        self.tick += 1    

        reward = 0
        if not self.done:    
            reward = -1
       
        # Handle action
        if action=='left':
            self.player_x = max(self.player_x-1, 0)
        elif action=='right':
            self.player_x = min(self.player_x+1, 9)
        elif action=='shoot':
            self.bullet_x = self.player_x
            self.bullet_y = 0
            self.bullet_on = True
            reward -= 10
        
        # Update bullet
        self.bullet_y += 1
        if self.bullet_y>9:
            self.bullet_on = False
        
        # Update enemy
        if not self.done:
            self.enemy_x += self.enemy_direction
            if self.enemy_x<0:
                self.enemy_x = 0
                self.enemy_y -= 1
                self.enemy_direction = -self.enemy_direction
            if self.enemy_x>9:
                self.enemy_x = 9
                self.enemy_y -= 1
                self.enemy_direction = -self.enemy_direction
                
        # Enemy won
        if self.enemy_y == 0 and not self.done:
            self.done = True
                
        # Enemy hit
        if self.bullet_x==self.enemy_x and self.bullet_y==self.enemy_y and self.bullet_on==True and not self.done:
            self.bullet_on = False
            self.done = True
            self.won = True
            reward += 100
        
        self.score += reward

        return (self.get_state(), reward, self.done)

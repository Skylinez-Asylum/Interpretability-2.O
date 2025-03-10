import sys
import pygame

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Colors
WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,0,255)

class Player:
    def __init__(self):
        self.rect = pygame.Rect(SCREEN_WIDTH//2 - 20, SCREEN_HEIGHT - 50, 40, 40) # rectangle for player
        self.speed = 5

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.left >0:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT] and self.rect.right < SCREEN_WIDTH:
            self.rect.x += self.speed

class Enemy:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x,y, 30,30)

    @staticmethod
    def create_new():
        # spawn enemies from top edges randomly?
        pass

# Alternatively, manage enemies as a list of rectangles.

enemies = []
# Add some initial enemies for testing:
for i in range(5):
    x = i * 100
    y = -30*i
    enemies.append(pygame.Rect(x,y,30,30))

bullets = []

player = Player()

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # shoot bullet from player's position
                bullet_x = player.rect.centerx -2
                bullet_y = player.rect.top
                bullets.append( pygame.Rect(bullet_x, bullet_y, 5,10) )

    # Update positions

    # Player movement
    player.update()

    # Move enemies downward
    for enemy in enemies:
        enemy.y += 2 # speed down
        if enemy.bottom > SCREEN_HEIGHT:  # off screen, remove?
            enemies.remove(enemy)

    # Move bullets upward
    for bullet in bullets[:]:
        bullet.y += -5 # moving up
        if bullet.top <0:
            bullets.remove(bullet)

    # Collision detection between bullets and enemies
    for bullet in bullets[:]: # iterate over copy to avoid issues during deletion
        for enemy in enemies[:]: # same here
            if bullet.colliderect(enemy):
                bullets.remove(bullet)
                enemies.remove(enemy)
                break  # exit inner loop, since bullet is gone

    # Draw everything
    screen.fill(WHITE)

    pygame.draw.rect(screen, BLUE, player.rect)

    for enemy in enemies:
        pygame.draw.rect(screen, RED, enemy)

    for bullet in bullets:
        pygame.draw.rect(screen, (0,255,0), bullet)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
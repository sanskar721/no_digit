import pygame
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('D:\Python\practise\mnist_model.h5')

# Pygame initialization
pygame.init()
screen = pygame.display.set_mode((280, 280))
pygame.display.set_caption("Digit Recognition")
clock = pygame.time.Clock()
drawing = False

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Font
font = pygame.font.SysFont(None, 36)

# Main loop
running = True
while running:
    screen.fill(WHITE)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pygame.draw.circle(screen, BLACK, event.pos, 10)

        # Predict the digit when the user presses the spacebar
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            pygame.image.save(screen, "digit.png")  # Save the drawn digit
            img = pygame.image.load("digit.png").convert("L")  # Convert to grayscale
            img = pygame.transform.scale(img, (28, 28))  # Resize to 28x28
            data = pygame.surfarray.array2d(img)  # Convert to array
            data = np.invert(data)  # Invert colors (black background, white digit)
            data = data.reshape(1, 784) / 255.0  # Flatten and normalize
            prediction = model.predict_classes(data)[0]  # Make prediction
            text = font.render(f"Prediction: {prediction}", True, BLACK)
            screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

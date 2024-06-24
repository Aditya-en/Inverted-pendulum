import numpy as np
import pygame
from environment import InvertedDoublePendulumEnv

def main():
    # Create the environment
    env = InvertedDoublePendulumEnv(gravity=8, friction=0.00001)
    
    # Reset the environment to start state
    obs = env.reset()
    
    done = False
    while not done:
        # Apply a random action
        action = env.action_space.sample()
        
        # Step the environment
        # obs, reward, done, info = env.step(np.array([0]))
        obs, reward, done, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Check if we should close the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
    
    env.close()

if __name__ == "__main__":
    main()

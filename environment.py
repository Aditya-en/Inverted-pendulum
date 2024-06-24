import gym
from gym import spaces
import numpy as np
import pygame
from pygame import gfxdraw

class InvertedDoublePendulumEnv(gym.Env):
    def __init__(self, gravity=9.8, friction=0.1, done_limit=100):
        super(InvertedDoublePendulumEnv, self).__init__()
        self.done_limit = done_limit
        self.gravity = gravity
        self.friction = friction

        self.dt = 0.02  # time step
        self.max_force = 10.0  # maximum force that can be applied

        # Define action and observation space
        self.action_space = spaces.Box(low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32)
        
        # The observation space includes the cart position, velocities, and angles of the two pendulums
        high = np.array([np.finfo(np.float32).max]*6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Initialize state
        self.state = None

        # Rendering
        self.screen = None
        self.screen_width = 800
        self.screen_height = 600
        self.cart_width = 50
        self.cart_height = 30
        self.pole_length = 150
        self.cart_y = self.screen_height / 2
        self.clock = None

    def reset(self):
        # Initialize the state to a small random value
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(6,))
        return self.state

    def step(self, action):
        # Unpack the state
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = self.state

        # Apply the action
        force = action[0]

        # Simplified physics for the double inverted pendulum
        # This part should include the actual physics equations which define the system dynamics
        # For simplicity, we are using a placeholder
        xacc = force - self.friction * x_dot
        theta1acc = self.gravity * np.sin(theta1) - self.friction * theta1_dot
        theta2acc = self.gravity * np.sin(theta2) - self.friction * theta2_dot

        # Update the state using Euler's method
        x_dot += self.dt * xacc
        x += self.dt * x_dot
        theta1_dot += self.dt * theta1acc
        theta1 += self.dt * theta1_dot
        theta2_dot += self.dt * theta2acc
        theta2 += self.dt * theta2_dot

        self.state = (x, x_dot, theta1, theta1_dot, theta2, theta2_dot)

        # Check if the episode is done
        done = bool(
            x < -self.done_limit or x > self.done_limit  # Cart position limits
            # or theta1 < -np.pi/2 or theta1 > np.pi/2  # Pendulum 1 angle limits
            # or theta2 < -np.pi/2 or theta2 > np.pi/2  # Pendulum 2 angle limits
        )

        # Calculate the reward
        reward = 1.0 if not done else 0.0

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Inverted Double Pendulum")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((255, 255, 255))

        x, _, theta1, _, theta2, _ = self.state

        # Convert state to screen coordinates
        cart_x = self.screen_width / 2 + x * 100
        pole1_end_x = cart_x + self.pole_length * np.sin(theta1)
        pole1_end_y = self.cart_y - self.pole_length * np.cos(theta1)
        pole2_end_x = pole1_end_x + self.pole_length * np.sin(theta2)
        pole2_end_y = pole1_end_y - self.pole_length * np.cos(theta2)

        # Draw cart
        pygame.draw.rect(self.screen, (0, 0, 0), [cart_x - self.cart_width / 2, self.cart_y - self.cart_height / 2, self.cart_width, self.cart_height])

        # Draw first pole
        pygame.draw.line(self.screen, (255, 0, 0), (cart_x, self.cart_y), (pole1_end_x, pole1_end_y), 5)

        # Draw second pole
        pygame.draw.line(self.screen, (0, 0, 255), (pole1_end_x, pole1_end_y), (pole2_end_x, pole2_end_y), 5)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

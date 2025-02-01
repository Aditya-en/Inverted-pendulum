import pygame
import math
import numpy as np

# -----------------------------
# Simulation parameters
# -----------------------------
# Screen parameters
WIDTH, HEIGHT = 1000, 600

# Physical parameters
M = 1000.0    # mass of the cart
m1 = 40.0     # mass of the first pendulum bob
m2 = 40.0     # mass of the second pendulum bob
L1 = 200.0    # length of the first pendulum rod (in pixels)
L2 = 200.0    # length of the second pendulum rod (in pixels)
g = 9.81      # gravitational acceleration

# Damping coefficients (tune these to change the dissipation rate)
b_cart = 100.0      # damping coefficient for cart velocity
b_theta1 = 80000.0     # damping coefficient for first pendulum angular velocity
b_theta2 = 80000.0     # damping coefficient for second pendulum angular velocity

# Integration parameters
dt = 0.02     # simulation time step

# Force limits (agent action)
FORCE_MAG = 10000.0  # magnitude of force applied when key is pressed

# -----------------------------
# Dynamics Functions
# -----------------------------
def dynamics(state, force):
    """
    Compute the state derivative for a cart with a double pendulum including damping.
    
    The state vector is:
      state = [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
      
    where:
      x          : horizontal position of the cart
      theta1   : angle of the first pendulum (from vertical, positive clockwise)
      theta2   : angle of the second pendulum (from vertical, positive clockwise)
      
    The damping is added as extra terms proportional to the velocities.
    """
    # Unpack state
    x, x_dot, theta1, theta1_dot, theta2, theta2_dot = state

    # Precompute sines and cosines
    s1 = math.sin(theta1)
    c1 = math.cos(theta1)
    s2 = math.sin(theta2)
    c2 = math.cos(theta2)
    s12 = math.sin(theta1 - theta2)
    c12 = math.cos(theta1 - theta2)

    # --- Construct the mass matrix M_mat and right-hand side vector b ---
    # Generalized coordinates: q = [x, theta1, theta2]
    # Mass matrix components:
    M11 = M + m1 + m2
    M12 = (m1 + m2) * L1 * c1
    M13 = m2 * L2 * c2

    M21 = M12
    M22 = (m1 + m2) * (L1 ** 2)
    M23 = m2 * L1 * L2 * c12

    M31 = M13
    M32 = M23
    M33 = m2 * (L2 ** 2)

    M_mat = np.array([
        [M11, M12, M13],
        [M21, M22, M23],
        [M31, M32, M33]
    ])

    # Original forcing terms from Lagrangian formulation:
    # Note: The terms involving squared angular velocities are due to the nonlinear dynamics.
    b1 = force - (m1 + m2) * L1 * s1 * (theta1_dot ** 2) - m2 * L2 * s2 * (theta2_dot ** 2)
    b2 = - m2 * L1 * L2 * s12 * (theta2_dot ** 2) - (m1 + m2) * g * L1 * s1
    b3 = m2 * L1 * L2 * s12 * (theta1_dot ** 2) - m2 * g * L2 * s2

    # --- Add damping contributions ---
    # For the cart: a friction force proportional to the velocity (x_dot)
    b1 -= b_cart * x_dot
    # For the pendulums: damping torques proportional to the angular velocities
    b2 -= b_theta1 * theta1_dot
    b3 -= b_theta2 * theta2_dot

    b_vec = np.array([b1, b2, b3])

    # Solve for the generalized accelerations: q_ddot = [x_ddot, theta1_ddot, theta2_ddot]
    try:
        q_ddot = np.linalg.solve(M_mat, b_vec)
    except np.linalg.LinAlgError:
        q_ddot = np.zeros(3)

    x_ddot, theta1_ddot, theta2_ddot = q_ddot

    return np.array([x_dot, x_ddot, theta1_dot, theta1_ddot, theta2_dot, theta2_ddot])

def rk4_step(state, force, dt):
    """Perform one RK4 integration step with given control force."""
    k1 = dynamics(state, force)
    k2 = dynamics(state + dt/2 * k1, force)
    k3 = dynamics(state + dt/2 * k2, force)
    k4 = dynamics(state + dt * k3, force)
    new_state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state

# -----------------------------
# Pygame Rendering Functions
# -----------------------------
def draw_system(screen, state):
    """
    Draw the cart, pendulums, and rail so that the system appears smaller
    and the rail is vertically centered. The cart's horizontal position
    reflects its simulation state so that it can move along the rail.
    """
    # Clear screen
    screen.fill((255, 255, 255))
    
    # --- Drawing parameters ---
    scale = 0.5  # Scale factor to shrink the system
    rail_y = HEIGHT // 2  # Place the rail at the vertical center
    
    # Unpack the simulation state.
    # state = [x, x_dot, theta1, theta1_dot, theta2, theta2_dot]
    x, _, theta1, _, theta2, _ = state

    # Compute the cart's horizontal position:
    # We use the simulation's x (scaled) added to the horizontal center of the screen.
    cart_x = WIDTH // 2 + int(x * scale)
    # Place the cart slightly above the rail.
    cart_y = rail_y - int(20 * scale)
    
    # --- Draw the rail ---
    # (Here the rail is drawn fixed in screen coordinates.)
    pygame.draw.line(screen, (0, 0, 0), (0, rail_y), (WIDTH, rail_y), 4)
    
    # --- Draw the cart ---
    cart_width, cart_height = 100 * scale, 40 * scale  # Scaled dimensions
    cart_rect = pygame.Rect(0, 0, cart_width, cart_height)
    cart_rect.center = (cart_x, cart_y)
    pygame.draw.rect(screen, (0, 128, 255), cart_rect)
    
    # --- Draw the pendulums ---
    # The pivot point for the pendulums is at the top-center of the cart.
    pivot = (cart_x, cart_y - cart_height // 2)
    
    # Use scaled lengths for drawing the pendulum rods.
    draw_L1 = L1 * scale
    draw_L2 = L2 * scale

    # Compute the positions of the pendulum bobs.
    # Note: Since y increases downward on the screen, we add to the y-coordinate.
    x1 = pivot[0] + draw_L1 * math.sin(theta1)
    y1 = pivot[1] + draw_L1 * math.cos(theta1)
    x2 = x1 + draw_L2 * math.sin(theta2)
    y2 = y1 + draw_L2 * math.cos(theta2)
    
    # Draw the rods (lines connecting the pivot to the first bob, and the first bob to the second bob)
    pygame.draw.line(screen, (0, 0, 0), pivot, (x1, y1), 3)
    pygame.draw.line(screen, (0, 0, 0), (x1, y1), (x2, y2), 3)
    
    # Draw the pendulum bobs as circles (scaling their size too)
    bob_radius = int(20 * scale)
    pygame.draw.circle(screen, (255, 0, 0), (int(x1), int(y1)), bob_radius)
    pygame.draw.circle(screen, (0, 255, 0), (int(x2), int(y2)), bob_radius)
    
    pygame.display.flip()


# -----------------------------
# Main Loop
# -----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Double Pendulum on a Cart with Damping (RL Environment)")
    clock = pygame.time.Clock()
    
    # Initial state:
    # x, x_dot, theta1, theta1_dot, theta2, theta2_dot
    # For the pendulums, an initial angle away from the bottom (0 radians) is used.
    state = np.array([0.0, 0.0, math.pi/2, 0.0, math.pi/2, 0.0])
    
    running = True
    while running:
        # --- Handle events and set the control force ---
        force = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Use key presses as a proxy for the RL agent's action:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            force = -FORCE_MAG
        elif keys[pygame.K_RIGHT]:
            force = FORCE_MAG
        
        # --- Update dynamics ---
        # For better numerical stability, perform several RK4 steps per frame.
        steps_per_frame = 5
        for _ in range(steps_per_frame):
            state = rk4_step(state, force, dt)
        
        # --- Render ---
        draw_system(screen, state)
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()

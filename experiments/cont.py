import numpy as np
import time
from inputimeout import inputimeout, TimeoutOccurred


def mutate(x, sigma, momentum):
    return x + sigma * np.random.randn(*x.shape) + momentum


x = np.random.randn(2)  # Initial individual
sigma = 0.1  # Initial mutation step-size
momentum = np.zeros(2)  # Initial momentum
alpha = 0.1  # Momentum factor
factor = 1.2  # Adaptation factor for sigma
alpha_increase = 1.1  # Factor to increase momentum influence
alpha_decay = 0.9  # Factor to decay momentum influence

print("Starting Evolution Strategy with Adaptive Momentum...")
print("Initial individual:", x)

try:
    while True:
        x_prime = mutate(x, sigma, alpha * momentum)
        print("\nNew individual after mutation:", x_prime)

        try:
            feedback = (
                inputimeout(
                    prompt="Provide feedback if needed ('n' for no, 'q' to quit, default is 'y'): ",
                    timeout=3,
                )
                .strip()
                .lower()
            )
        except TimeoutOccurred:
            feedback = "y"  # Assume 'y' if no input is provided within 1 second

        if feedback == "q":
            print("Exiting...")
            break
        elif feedback == "n":
            sigma /= factor
            alpha *= alpha_decay  # Decay the influence of momentum
        else:  # Treat any input other than 'n' or 'q' as positive feedback
            momentum = x_prime - x  # Update momentum vector
            x = x_prime
            sigma *= factor
            alpha *= alpha_increase  # Increase the influence of momentum

        print("Current individual:", x)
        print("Current step-size (sigma):", sigma)
        print("Current momentum:", momentum)
        print("Current momentum factor (alpha):", alpha)

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")

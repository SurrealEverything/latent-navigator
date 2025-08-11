import numpy as np


def mutate(x, sigma, momentum):
    return x + sigma * np.random.randn(*x.shape) + momentum


x = np.random.randn(2)  # Initial individual
sigma = 0.1  # Initial mutation step-size
momentum = np.zeros(2)  # Initial momentum
alpha = 0.1  # Initial momentum factor
factor = 1.2  # Adaptation factor for sigma
alpha_increase = 1.1  # Factor to increase momentum influence
alpha_decay = 0.9  # Factor to decay momentum influence

print("Starting Evolution Strategy with Adaptive Momentum...")
print("Initial individual:", x)

while True:
    x_prime = mutate(x, sigma, alpha * momentum)
    print("\nNew individual after mutation:", x_prime)

    feedback = (
        input(
            "Is the new individual good? (Type 'y' for yes, 'n' for no, 'q' to quit): "
        )
        .strip()
        .lower()
    )
    if feedback == "q":
        print("Exiting...")
        break
    elif feedback == "y":
        momentum = x_prime - x  # Update momentum vector
        x = x_prime
        sigma *= factor
        alpha *= alpha_increase  # Increase the influence of momentum
    elif feedback == "n":
        sigma /= factor
        alpha *= alpha_decay  # Decay the influence of momentum
    else:
        print("Invalid input. Please type 'y', 'n', or 'q'.")

    print("Current individual:", x)
    # print("Current step-size (sigma):", sigma)
    # print("Current momentum:", momentum)
    # print("Current momentum factor (alpha):", alpha)

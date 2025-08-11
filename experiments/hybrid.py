import numpy as np
from inputimeout import inputimeout, TimeoutOccurred


def sample_solution(mean, std_dev):
    return np.random.normal(mean, std_dev)


def update_parameters(feedback, sample, mean, std_dev, momentum):
    if feedback == "n":
        # Negative feedback: increase std_dev to explore more
        std_dev = np.minimum(std_dev * (1 + adapt_rate), max_std_dev)
    else:
        # Positive feedback: move mean towards sample, decrease std_dev
        new_mean = mean + adapt_rate * (sample - mean) + momentum
        new_std_dev = std_dev * (1 - adapt_rate)
        mean = new_mean
        std_dev = np.maximum(
            new_std_dev, 0.1
        )  # Prevent std_dev from becoming too small
        momentum = adapt_momentum * (sample - mean)  # Update momentum with decay
    return mean, std_dev, momentum


mean = np.zeros(2)  # Initial mean of the distribution
std_dev = np.ones(2)  # Initial standard deviation
momentum = np.zeros(2)  # Initial momentum
adapt_rate = 0.1  # Rate of adaptation for mean and std_dev
adapt_momentum = 0.1  # Momentum adaptation factor
max_std_dev = 2.0  # Maximum standard deviation to prevent explosion

print("Starting Hybrid Optimization with Adaptive Momentum and CEM...")
print("Initial distribution mean:", mean, "Std Dev:", std_dev)

try:
    while True:
        sample = sample_solution(mean, std_dev)
        print("\nSampled individual:", sample)

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
            feedback = "y"  # Default to 'y' if no input is provided within 3 seconds

        if feedback == "q":
            print("Exiting...")
            break

        mean, std_dev, momentum = update_parameters(
            feedback, sample, mean, std_dev, momentum
        )
        print("Updated distribution mean:", mean, "Std Dev:", std_dev)
        print("Current momentum:", momentum)

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")

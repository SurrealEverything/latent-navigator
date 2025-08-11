import numpy as np
import time
from inputimeout import inputimeout, TimeoutOccurred


def sample_solution(mean, std_dev):
    return np.random.normal(mean, std_dev)


mean = np.zeros(2)  # Initial mean for the distribution
std_dev = np.ones(2)  # Initial standard deviation
adapt_rate = 0.1  # Rate at which to adapt mean and std_dev based on feedback

print("Starting Cross-Entropy Method Optimization...")
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
            feedback = "y"  # Assume 'y' if no input is provided within 1 second

        if feedback == "q":
            print("Exiting...")
            break
        elif feedback == "n":
            # Update distribution to search away from the current sample
            std_dev += adapt_rate * (np.abs(sample - mean))
        else:  # Treat any input other than 'n' or 'q' as positive feedback
            # Update distribution to focus more around the current sample
            new_mean = mean + adapt_rate * (sample - mean)
            new_std_dev = std_dev - adapt_rate * (std_dev - np.abs(sample - mean))
            std_dev = np.clip(
                new_std_dev, 0.1, None
            )  # Prevent std_dev from going to 0 or negative
            mean = new_mean

        print("Updated distribution mean:", mean, "Std Dev:", std_dev)

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")

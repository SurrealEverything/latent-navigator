import numpy as np
from inputimeout import inputimeout, TimeoutOccurred


def sample_solution(mean, std_dev):
    return np.random.normal(mean, std_dev)


mean = np.zeros(2)  # Initial mean for the distribution
std_dev = np.ones(2)  # Initial standard deviation
adapt_rate = 0.1  # Rate at which to adapt mean and std_dev
max_std_dev = 2.0  # Max standard deviation to prevent explosion

print("Starting Cross-Entropy Method Optimization...")
print("Initial distribution mean:", mean, "Std Dev:", std_dev)

try:
    while True:
        sample = sample_solution(mean, std_dev)
        print("\nSampled individual:", sample)

        try:
            feedback = (
                inputimeout(
                    prompt="Provide feedback if needed ('n' for no, 'q' to quit, default is 'n'): ",
                    timeout=3,
                )
                .strip()
                .lower()
            )
        except TimeoutOccurred:
            feedback = "n"  # Assume 'n' if no input is provided within 3 seconds

        if feedback == "q":
            print("Exiting...")
            break
        elif feedback == "n":
            std_dev = np.minimum(std_dev * (1 + adapt_rate), max_std_dev)
        else:
            new_mean = mean + adapt_rate * (sample - mean)
            new_std_dev = std_dev * (1 - adapt_rate)
            mean = new_mean
            std_dev = np.maximum(
                new_std_dev, 0.1
            )  # Prevent std_dev from going to 0 or negative

        print("Updated distribution mean:", mean, "Std Dev:", std_dev)

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")

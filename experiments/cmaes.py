import cma
import numpy as np


def get_user_feedback(x):
    print("\nCurrent individual:", x)
    feedback = (
        input("Is this individual good? (Type 'y' for yes, 'n' for no, 'q' to quit): ")
        .strip()
        .lower()
    )
    return feedback


# Initial parameters
dimension = 2  # Dimension of the problem
initial_solution = np.zeros(dimension)  # Starting point

# Configure the CMA-ES optimizer
es = cma.CMAEvolutionStrategy(initial_solution, 0.5, {"verbose": -9})

print("Starting CMA-ES Optimization...")

# Optimization loop
while not es.stop():
    solutions = es.ask()  # Generate new candidate solutions
    evaluations = []

    for solution in solutions:
        feedback = get_user_feedback(solution)
        if feedback == "q":
            print("Exiting...")
            break
        evaluations.append(
            1.0 if feedback == "y" else 0.0
        )  # Binary fitness: 1 for good, 0 for bad

    if feedback == "q":
        break

    es.tell(solutions, evaluations)  # Update the internal model
    es.disp()  # Optionally display the internal state

print("Optimization completed.")

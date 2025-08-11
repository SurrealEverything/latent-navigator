import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import cv2
from pynput import keyboard

# Load the diffusion model
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")
prompt = "Shrek underwear"
# prompt = "A cinematic shot of a baby hamster wearing an intricate italian priest robe."
# prompt = "Ninja saves Sensei by defeating the Ranx, I-Ninja video game"

# Parameters
pop_size = 10
num_generations = 50
mutation_rate = 0.1
min_mutation_rate = 0.01
max_mutation_rate = 0.3
elite_fraction = 0.2
noise_shape = (1, 64, 64, 4)
key_feedback = None


def initialize_population(pop_size, shape):
    return [np.random.normal(0, 1, shape) for _ in range(pop_size)]


def mutate(individual, mutation_rate):
    mutation = np.random.normal(0, mutation_rate, individual.shape)
    return individual + mutation


def crossover(parent1, parent2):
    mask = np.random.rand(*parent1.shape) > 0.5
    child = np.where(mask, parent1, parent2)
    return child


def display_image(noise):
    noise_tensor = torch.tensor(noise, dtype=torch.float16, device="cuda").permute(
        0, 3, 1, 2
    )
    image = pipe(
        prompt=prompt, num_inference_steps=4, guidance_scale=0.0, latents=noise_tensor
    ).images[0]
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV
    cv2.imshow("Optimized Image", open_cv_image)
    cv2.waitKey(1)  # Add a short delay to update the image


def on_press(key):
    global key_feedback
    try:
        if key.char == "y":
            key_feedback = 1  # Positive feedback
        elif key.char == "n":
            key_feedback = 0  # Negative feedback
        elif key.char == "q":
            key_feedback = "q"  # Quit signal
    except AttributeError:
        pass


def tournament_selection(population, fitness_scores, k=3):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(len(population), k, replace=False)
        best = max(participants, key=lambda x: fitness_scores[x])
        selected.append(population[best])
    return selected


# Initialize population
population = initialize_population(pop_size, noise_shape)
elite_size = int(pop_size * elite_fraction)

print("Starting Genetic Algorithm Optimization...")
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    for generation in range(num_generations):
        fitness_scores = []

        for individual in population:
            display_image(individual)
            key_feedback = None

            while key_feedback is None:
                cv2.waitKey(1)  # Wait for user input

            if key_feedback == "q":
                print("Exiting...")
                raise KeyboardInterrupt
            fitness_scores.append((key_feedback, individual))

        # Sort by fitness score
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        elite_individuals = [x[1] for x in fitness_scores[:elite_size]]

        # Create new population
        new_population = elite_individuals.copy()

        # Use tournament selection to create mating pool
        mating_pool = tournament_selection(population, [x[0] for x in fitness_scores])

        while len(new_population) < pop_size:
            parent_indices = np.random.choice(len(mating_pool), 2, replace=False)
            parent1, parent2 = (
                mating_pool[parent_indices[0]],
                mating_pool[parent_indices[1]],
            )
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        # Adjust mutation rate dynamically
        if generation > 0 and fitness_scores[0][0] == fitness_scores[-1][0]:
            mutation_rate = min(max_mutation_rate, mutation_rate * 1.1)
        else:
            mutation_rate = max(min_mutation_rate, mutation_rate * 0.9)

        print(f"Generation {generation + 1} complete")

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")
finally:
    cv2.destroyAllWindows()  # Ensure all windows are closed when the script terminates
    listener.stop()

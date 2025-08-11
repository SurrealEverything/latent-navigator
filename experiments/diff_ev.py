import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import cv2

# Load the diffusion model
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."


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
        prompt=prompt, num_inference_steps=1, guidance_scale=0.0, latents=noise_tensor
    ).images[0]
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV
    cv2.imshow("Optimized Image", open_cv_image)


# Parameters
pop_size = 10
num_generations = 50
mutation_rate = 0.1
elite_fraction = 0.2

# Initialize population
noise_shape = (1, 64, 64, 4)
population = initialize_population(pop_size, noise_shape)
elite_size = int(pop_size * elite_fraction)

# Main optimization loop
print("Starting Genetic Algorithm Optimization...")
try:
    for generation in range(num_generations):
        fitness_scores = []

        for individual in population:
            display_image(individual)
            key = cv2.waitKey(3000) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                raise KeyboardInterrupt
            elif key == ord("y"):
                fitness_scores.append((individual, 1))
            else:
                fitness_scores.append((individual, 0))

        # Sort by fitness score
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        elite_individuals = [ind for ind, score in fitness_scores[:elite_size]]

        # Create new population
        new_population = elite_individuals.copy()

        while len(new_population) < pop_size:
            parent_indices = np.random.choice(len(elite_individuals), 2, replace=False)
            parent1, parent2 = (
                elite_individuals[parent_indices[0]],
                elite_individuals[parent_indices[1]],
            )
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        print(f"Generation {generation + 1}: Best Fitness = {fitness_scores[0][1]}")

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")
finally:
    cv2.destroyAllWindows()  # Ensure all windows are closed when the script terminates

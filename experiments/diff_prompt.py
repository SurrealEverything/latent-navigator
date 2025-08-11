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
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

# Parameters
pop_size = 10
num_generations = 50
mutation_rate = 0.1
min_mutation_rate = 0.01
max_mutation_rate = 0.3
elite_fraction = 0.2
key_feedback = None


def initialize_population(pipe, prompt, pop_size):
    prompt_embeds = []
    pooled_prompt_embeds = []
    for _ in range(pop_size):
        debug = pipe.encode_prompt(prompt)
        print(len(debug), debug)
        prompt_embed, pooled_embed = debug
        prompt_embeds.append(prompt_embed.clone().detach().cpu().numpy())
        pooled_prompt_embeds.append(pooled_embed.clone().detach().cpu().numpy())
    return prompt_embeds, pooled_prompt_embeds


def mutate(individual, mutation_rate):
    mutation = np.random.normal(0, mutation_rate, individual.shape)
    return individual + mutation


def crossover(parent1, parent2):
    mask = np.random.rand(*parent1.shape) > 0.5
    child = np.where(mask, parent1, parent2)
    return child


def display_image(prompt_embedding, pooled_embedding):
    prompt_embedding_tensor = torch.tensor(
        prompt_embedding, dtype=torch.float16, device="cuda"
    )
    pooled_embedding_tensor = torch.tensor(
        pooled_embedding, dtype=torch.float16, device="cuda"
    )
    image = pipe(
        prompt=None,
        num_inference_steps=1,
        guidance_scale=5.0,
        prompt_embeds=prompt_embedding_tensor,
        pooled_prompt_embeds=pooled_embedding_tensor,
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
prompt_population, pooled_population = initialize_population(pipe, prompt, pop_size)
elite_size = int(pop_size * elite_fraction)

print("Starting Genetic Algorithm Optimization...")
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    for generation in range(num_generations):
        fitness_scores = []

        for prompt_individual, pooled_individual in zip(
            prompt_population, pooled_population
        ):
            display_image(prompt_individual, pooled_individual)
            key_feedback = None

            while key_feedback is None:
                cv2.waitKey(1)  # Wait for user input

            if key_feedback == "q":
                print("Exiting...")
                raise KeyboardInterrupt
            fitness_scores.append(
                (key_feedback, (prompt_individual, pooled_individual))
            )

        # Sort by fitness score
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        elite_individuals = [x[1] for x in fitness_scores[:elite_size]]

        # Create new population
        new_prompt_population = [x[0] for x in elite_individuals]
        new_pooled_population = [x[1] for x in elite_individuals]

        # Use tournament selection to create mating pool
        mating_pool = tournament_selection(
            elite_individuals, [x[0] for x in fitness_scores]
        )

        while len(new_prompt_population) < pop_size:
            parent_indices = np.random.choice(len(mating_pool), 2, replace=False)
            parent1, parent2 = (
                mating_pool[parent_indices[0]],
                mating_pool[parent_indices[1]],
            )
            child_prompt = crossover(parent1[0], parent2[0])
            child_prompt = mutate(child_prompt, mutation_rate)
            new_prompt_population.append(child_prompt)

            child_pooled = crossover(parent1[1], parent2[1])
            child_pooled = mutate(child_pooled, mutation_rate)
            new_pooled_population.append(child_pooled)

        prompt_population = new_prompt_population
        pooled_population = new_pooled_population

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

import os
import string
import time

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
import cv2
from pynput import keyboard
from evolutionary_prompt_embedding.image_creation import SDXLPromptEmbeddingImageCreator
from evolutionary_prompt_embedding.variation import (
    UniformGaussianMutatorArguments,
    PooledUniformGaussianMutator,
    PooledArithmeticCrossover,
)
from evolutionary_prompt_embedding.value_ranges import (
    SDXLTurboEmbeddingRange,
    SDXLTurboPooledEmbeddingRange,
)
from evolutionary.evolutionary_selectors import TournamentSelector
from evolutionary.evolution_base import SolutionCandidate

# Load the diffusion model
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")
prompt = "Green monster surprised by a jump scare in a horror movie"

# Parameters
pop_size = 5
num_generations = 5000
mutation_rate = 0.1
crossover_proportion = 0.8
crossover_rate = 0.9
inference_steps = 4
key_feedback = None

# Generate timestamp
current_timestamp = int(time.time())

# Format the prompt
formatted_prompt = ''.join(e.lower() if e.isalnum() else '_' for e in prompt if e.isalnum() or e.isspace()).strip()

# Generate folder name
folder_name = f"{current_timestamp}_{formatted_prompt}"
folder_path = os.path.join("saved_images", folder_name)

# Define min/max values for the prompt embeddings
embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()

# Initialize saving flag
image_to_save = None

# Initialize components for the genetic algorithm
creator = SDXLPromptEmbeddingImageCreator(batch_size=1, inference_steps=inference_steps)
crossover = PooledArithmeticCrossover(
    interpolation_weight=0.5,
    interpolation_weight_pooled=0.5,
    proportion=crossover_proportion,
    proportion_pooled=crossover_proportion,
)
mutation_arguments = UniformGaussianMutatorArguments(
    mutation_rate=mutation_rate,
    mutation_strength=2,
    clamp_range=(embedding_range.minimum, embedding_range.maximum),
)
mutation_arguments_pooled = UniformGaussianMutatorArguments(
    mutation_rate=mutation_rate,
    mutation_strength=0.4,
    clamp_range=(pooled_embedding_range.minimum, pooled_embedding_range.maximum),
)
mutator = PooledUniformGaussianMutator(mutation_arguments, mutation_arguments_pooled)
selector = TournamentSelector(tournament_size=3)

# Initialize population from prompt
initial_argument = creator.arguments_from_prompt(prompt)
population = [SolutionCandidate(initial_argument, None) for _ in range(pop_size)]

# Initialize elite storage
stored_elites = []

print("Starting Genetic Algorithm Optimization...")


# Function to display image from prompt embedding
def display_image(prompt_embed_data):
    noise_tensor = (
        prompt_embed_data.prompt_embeds.clone()
        .detach()
        .requires_grad_(False)
        .to("cuda")
    )
    pooled_tensor = (
        prompt_embed_data.pooled_prompt_embeds.clone()
        .detach()
        .requires_grad_(False)
        .to("cuda")
    )
    image = pipe(
        prompt_embeds=noise_tensor,
        pooled_prompt_embeds=pooled_tensor,
        num_inference_steps=inference_steps,
        guidance_scale=0.0,
    ).images[0]
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV

    # Create a copy for display that includes instructions
    display_image = open_cv_image.copy()

    # Instructions for user input, split across multiple lines
    instructions = [
        "Press 'Y' for Yes",
        "Press 'N' for No",
        "Press 'S' to Save",
        "Press 'Q' to Quit"
    ]
    for i, line in enumerate(instructions):
        cv2.putText(display_image, line, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Configure window to remove the clickable portion on top and set its size
    window_name = "Optimized Image"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(window_name, display_image)
    cv2.resizeWindow(window_name, open_cv_image.shape[1], open_cv_image.shape[0] + 120)  # Add extra space for instructions
    cv2.waitKey(1)  # Add a short delay to update the image

    return open_cv_image




# Listener for keyboard inputs
def on_press(key):
    global key_feedback, image_to_save
    try:
        if key.char == "y":
            key_feedback = 1  # Positive feedback
        elif key.char == "n":
            key_feedback = 0  # Negative feedback
        elif key.char == "s":
            image_to_save = True  # Set flag to save the image
        elif key.char == "q":
            key_feedback = "q"  # Quit signal
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Run the genetic algorithm
try:
    for generation in range(num_generations):
        fitness_scores = []

        for individual in population:
            open_cv_image = display_image(individual.arguments)
            key_feedback = None

            while key_feedback is None:
                cv2.waitKey(1)  # Wait for user input

                if image_to_save:
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    image_path = os.path.join(folder_path, f"generation{generation + 1}_individual_{population.index(individual) + 1}.png")
                    cv2.imwrite(image_path, open_cv_image)  # Save the current image
                    image_to_save = None  # Reset the flag

            if key_feedback == "q":
                print("Exiting...")
                raise KeyboardInterrupt

            individual.fitness = key_feedback
            fitness_scores.append((key_feedback, individual))

        elite_individuals = [x[1] for x in fitness_scores if x[0] == 1]

        # If no individual passes, reuse last elites
        if len(elite_individuals) == 0:
            if len(stored_elites) == 0:
                elite_individuals = [
                    SolutionCandidate(creator.arguments_from_prompt(prompt), None)
                    for _ in range(pop_size)
                ]
            else:
                elite_individuals = stored_elites

        stored_elites = elite_individuals  # Store the elites

        # Create new population
        new_population = elite_individuals.copy()

        # Use tournament selection to create mating pool
        mating_pool = [
            selector.select([x[1] for x in fitness_scores]) for _ in range(pop_size)
        ]

        while len(new_population) < pop_size:
            parent_indices = np.random.choice(len(mating_pool), 2, replace=False)
            parent1, parent2 = (
                mating_pool[parent_indices[0]],
                mating_pool[parent_indices[1]],
            )
            if np.random.rand() < crossover_rate:
                child = crossover.crossover(parent1.arguments, parent2.arguments)
            else:
                child = parent1.arguments
            child = mutator.mutate(child)
            new_population.append(SolutionCandidate(child, None))

        population = new_population

        print(f"Generation {generation + 1} complete")

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")
finally:
    cv2.destroyAllWindows()  # Ensure all windows are closed when the script terminates
    listener.stop()

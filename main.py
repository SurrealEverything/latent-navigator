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

# Define min/max values for the prompt embeddings
embedding_range = SDXLTurboEmbeddingRange()
pooled_embedding_range = SDXLTurboPooledEmbeddingRange()

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
    cv2.imshow("Optimized Image", open_cv_image)
    cv2.waitKey(1)  # Add a short delay to update the image


# Listener for keyboard inputs
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


listener = keyboard.Listener(on_press=on_press)
listener.start()

# Run the genetic algorithm
try:
    for generation in range(num_generations):
        fitness_scores = []

        for individual in population:
            display_image(individual.arguments)
            key_feedback = None

            while key_feedback is None:
                cv2.waitKey(1)  # Wait for user input

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

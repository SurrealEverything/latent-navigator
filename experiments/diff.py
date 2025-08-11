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


def sample_population(mean, std_dev, pop_size):
    population = np.random.normal(mean, std_dev, size=(pop_size,) + mean.shape)
    population = np.clip(population, -1, 1)  # Ensure values are within bounds
    return population


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


def update_distribution(elite_samples):
    new_mean = np.mean(elite_samples, axis=0)
    new_std_dev = np.std(elite_samples, axis=0)
    return new_mean, new_std_dev


# Parameters
pop_size = 10
num_generations = 50
elite_fraction = 0.2
noise_shape = (1, 64, 64, 4)

# Initialize mean and std_dev
mean = np.zeros(noise_shape)
std_dev = np.ones(noise_shape)
elite_size = int(pop_size * elite_fraction)

# Main optimization loop
print("Starting Cross-Entropy Method Optimization...")
try:
    for generation in range(num_generations):
        population = sample_population(mean, std_dev, pop_size)
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
        elite_samples = np.array([ind for ind, score in fitness_scores[:elite_size]])

        # Update mean and std_dev based on elite samples
        if len(elite_samples) > 0:
            mean, std_dev = update_distribution(elite_samples)

        print(f"Generation {generation + 1}: Best Fitness = {fitness_scores[0][1]}")

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")
finally:
    cv2.destroyAllWindows()  # Ensure all windows are closed when the script terminates

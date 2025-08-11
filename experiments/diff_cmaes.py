import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import cv2
import cma

# Load the diffusion model
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")
prompt = ""#A cinematic shot of a baby racoon wearing an intricate italian priest robe."


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


def regularize_noise(noise, original_mean, original_std, reg_factor=0.1):
    return reg_factor * np.linalg.norm(noise - original_mean) / original_std


# Initialize parameters
noise_shape = (1, 64, 64, 4)
init_mean = np.zeros(noise_shape)
init_std = 1.0
reg_factor = 0.1

# Generate initial random noise to use as the baseline distribution
initial_noise = np.random.normal(init_mean, init_std, size=noise_shape)

# Initialize CMA-ES
es = cma.CMAEvolutionStrategy(init_mean.flatten(), init_std)

print("Starting CMA-ES Optimization with Regularization...")
try:
    while not es.stop():
        solutions = es.ask()
        fitness_scores = []

        for solution in solutions:
            solution_reshaped = solution.reshape(noise_shape)
            display_image(solution_reshaped)
            key = cv2.waitKey(3000) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                raise KeyboardInterrupt
            elif key == ord("y"):
                fitness = 1 - regularize_noise(
                    solution_reshaped, initial_noise, init_std, reg_factor
                )
                fitness_scores.append(fitness)
            else:
                fitness_scores.append(0)

        es.tell(
            solutions, [-score for score in fitness_scores]
        )  # CMA-ES minimizes, so invert the scores
        es.disp()

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")
finally:
    cv2.destroyAllWindows()  # Ensure all windows are closed when the script terminates

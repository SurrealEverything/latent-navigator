import numpy as np
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import cv2
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from pynput import keyboard

# Load the diffusion model
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

# Initialize parameters
noise_shape = (1, 64, 64, 4)
key_feedback = None


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
    cv2.waitKey(1)  # Add a short delay to update the image


def on_press(key):
    global key_feedback
    try:
        if key.char == "y":
            key_feedback = -1  # Maximizing the fitness
        elif key.char == "n":
            key_feedback = 0  # Minimizing the fitness
        elif key.char == "q":
            key_feedback = "q"  # Quit signal
    except AttributeError:
        pass


# Define the search space
space = {
    "mean": [
        hp.uniform(f"mean{i}", -1, 1)
        for i in range(noise_shape[1] * noise_shape[2] * noise_shape[3])
    ],
    "std": [
        hp.uniform(f"std{i}", 0, 1)
        for i in range(noise_shape[1] * noise_shape[2] * noise_shape[3])
    ],
}


def objective(params):
    global key_feedback
    key_feedback = None

    mean = np.array(params["mean"]).reshape(noise_shape)
    std = np.array(params["std"]).reshape(noise_shape)
    noise = np.random.normal(mean, std)

    display_image(noise)

    while key_feedback is None:
        cv2.waitKey(1)  # Wait for user input

    if key_feedback == "q":
        print("Exiting...")
        raise KeyboardInterrupt
    elif key_feedback == -1:
        return {"loss": -1, "status": STATUS_OK}  # Maximizing the fitness
    else:
        return {"loss": 0, "status": STATUS_OK}  # Minimizing the fitness


# Initialize Trials object to store the results
trials = Trials()

print("Starting Bayesian Optimization with Hyperopt...")
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials
    )
    print(f"Best parameters found: {best}")

except KeyboardInterrupt:
    print("Manual interrupt, stopping...")
finally:
    cv2.destroyAllWindows()  # Ensure all windows are closed when the script terminates
    listener.stop()

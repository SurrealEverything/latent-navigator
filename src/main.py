import os
import time

import cv2
import numpy as np
import yaml
from pynput import keyboard

from evolutionary.evolution_base import SolutionCandidate
from evolutionary.evolutionary_selectors import TournamentSelector
from evolutionary_prompt_embedding.image_creation import SDXLPromptEmbeddingImageCreator
from evolutionary_prompt_embedding.value_ranges import (
    SDXLTurboEmbeddingRange,
    SDXLTurboPooledEmbeddingRange,
)
from evolutionary_prompt_embedding.variation import (
    PooledArithmeticCrossover,
    PooledUniformGaussianMutator,
    UniformGaussianMutatorArguments,
)


class GeneticAlgorithm:
    def __init__(self):
        self.key_feedback = None
        self.image_to_save = False
        self.load_config()
        self.setup_components()

    def load_config(self):
        with open("config.yaml", "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def setup_components(self):
        self.creator = SDXLPromptEmbeddingImageCreator(
            batch_size=1,
            inference_steps=self.config["inference_steps"],
            deterministic=self.config["deterministic"],
        )
        self.crossover = PooledArithmeticCrossover(
            interpolation_weight=0.5,
            interpolation_weight_pooled=0.5,
            proportion=self.config["crossover_proportion"],
            proportion_pooled=self.config["crossover_proportion"],
        )
        self.mutator = PooledUniformGaussianMutator(
            UniformGaussianMutatorArguments(
                mutation_rate=self.config["mutation_rate"],
                mutation_strength=2,
                clamp_range=(
                    SDXLTurboEmbeddingRange().minimum,
                    SDXLTurboEmbeddingRange().maximum,
                ),
            ),
            UniformGaussianMutatorArguments(
                mutation_rate=self.config["mutation_rate"],
                mutation_strength=0.4,
                clamp_range=(
                    SDXLTurboPooledEmbeddingRange().minimum,
                    SDXLTurboPooledEmbeddingRange().maximum,
                ),
            ),
        )
        self.selector = TournamentSelector(tournament_size=3)

    def on_press(self, key):
        """Handle keyboard inputs."""
        try:
            if key.char == "y":
                self.key_feedback = 1
            elif key.char == "n":
                self.key_feedback = 0
            elif key.char == "s":
                self.image_to_save = True
            elif key.char == "q":
                self.key_feedback = "q"
        except AttributeError:
            pass

    def start_keyboard_listener(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        return listener

    # Function to display image from prompt embedding
    def display_image(self, prompt_embed_data):
        image = self.creator.create_solution(prompt_embed_data).result.images[0]

        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[
            :, :, ::-1
        ].copy()  # Convert RGB to BGR for OpenCV

        # Create a copy for display that includes instructions
        display_image = open_cv_image.copy()

        # Instructions for user input, split across multiple lines
        instructions = [
            "Press 'Y' for Yes",
            "Press 'N' for No",
            "Press 'S' to Save",
            "Press 'Q' to Quit",
        ]
        for i, line in enumerate(instructions):
            cv2.putText(
                display_image,
                line,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Configure window to remove the clickable portion on top and set its size
        window_name = "Optimized Image"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(window_name, display_image)
        cv2.resizeWindow(
            window_name, open_cv_image.shape[1], open_cv_image.shape[0] + 120
        )  # Add extra space for instructions
        cv2.waitKey(1)  # Add a short delay to update the image

        return open_cv_image

    def run(self):
        """Run the genetic algorithm."""
        prompt_input = input(
            "Enter a prompt for the genetic algorithm or press Enter to use the default: "
        ).strip()
        prompt = prompt_input if prompt_input else self.config["default_prompt"]

        current_timestamp = int(time.time())
        formatted_prompt = "".join(
            e.lower() if e.isalnum() else "_"
            for e in prompt
            if e.isalnum() or e.isspace()
        ).strip()
        folder_path = os.path.join(
            "saved_images", f"{current_timestamp}_{formatted_prompt}"
        )

        population = [
            SolutionCandidate(self.creator.arguments_from_prompt(prompt), None)
            for _ in range(self.config["pop_size"])
        ]
        stored_elites = []

        print("Starting Genetic Algorithm Optimization...")
        listener = self.start_keyboard_listener()

        try:
            for generation in range(self.config["num_generations"]):
                fitness_scores = []

                for individual in population:
                    open_cv_image = self.display_image(individual.arguments)
                    self.key_feedback = None

                    while self.key_feedback is None:
                        cv2.waitKey(1)

                        if self.image_to_save:
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)
                            image_path = os.path.join(
                                folder_path,
                                f"generation{generation + 1}_individual_{population.index(individual) + 1}.png",
                            )
                            cv2.imwrite(image_path, open_cv_image)
                            self.image_to_save = False

                    if self.key_feedback == "q":
                        print("Exiting...")
                        raise KeyboardInterrupt

                    individual.fitness = self.key_feedback
                    fitness_scores.append((self.key_feedback, individual))

                elite_individuals = [x[1] for x in fitness_scores if x[0] == 1]
                stored_elites = (
                    elite_individuals if elite_individuals else stored_elites
                )
                new_population = stored_elites.copy()

                mating_pool = [
                    self.selector.select([x[1] for x in fitness_scores])
                    for _ in range(self.config["pop_size"])
                ]
                while len(new_population) < self.config["pop_size"]:
                    parent_indices = np.random.choice(
                        len(mating_pool), 2, replace=False
                    )
                    parent1, parent2 = (
                        mating_pool[parent_indices[0]],
                        mating_pool[parent_indices[1]],
                    )
                    child = (
                        self.crossover.crossover(parent1.arguments, parent2.arguments)
                        if np.random.rand() < self.config["crossover_rate"]
                        else parent1.arguments
                    )
                    new_population.append(
                        SolutionCandidate(self.mutator.mutate(child), None)
                    )
                population = new_population

                print(f"Generation {generation + 1} complete")

        except KeyboardInterrupt:
            print("Manual interrupt, stopping...")
        finally:
            cv2.destroyAllWindows()
            listener.stop()


if __name__ == "__main__":
    GeneticAlgorithm().run()

# Latent Navigator

Interactive, human-in-the-loop evolutionary search through the SDXL-turbo latent space using prompt embeddings. Provide simple Y/N feedback to steer generations in real time and optionally save images you like.

Built on top of Diffusers and a lightweight evolutionary toolkit included in this repo.

> Default model: `stabilityai/sdxl-turbo`

## Features

- Interactive evolution with binary feedback (Yes/No)
- Prompt-embedding operators: arithmetic crossover and Gaussian mutation on SDXL prompt embeddings (incl. pooled embeddings)
- Elitism and tournament selection
- On-the-fly saving with `S` key to `saved_images/`
- Config-first via `config.yaml`

## Quick start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch (pick the correct CUDA/CPU build):
# See https://pytorch.org/get-started/locally/ for the right command, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install core requirements
pip install -r requirements.txt

# YAML parser used by src/main.py
pip install pyyaml
```

Notes:

- A GPU is strongly recommended for SDXL; CPU will be slow.
- On first run, Diffusers downloads `stabilityai/sdxl-turbo` weights.

### 2) Run

Project uses a `src/` layout. Easiest is to set `PYTHONPATH` for the invocation:

```bash
PYTHONPATH=src python src/main.py
```

Optionally install the local packages and extras in editable mode:

```bash
pip install -e .
```

## Usage (interactive controls)

When the window opens, use the keyboard:

- Y — mark image as good (fitness = 1)
- N — mark image as bad (fitness = 0)
- S — save the current image to `saved_images/<timestamp>_<prompt>/`
- Q — quit

The loop runs for `num_generations` or until quit. Elites are kept if any; offspring are created via tournament selection, arithmetic crossover, and Gaussian mutation on prompt embeddings.

## Configuration

Tunables in `config.yaml`:

```yaml
default_prompt: "Green monster surprised by a jump scare in a horror movie"
pop_size: 5
num_generations: 5000
mutation_rate: 0.1
crossover_proportion: 0.8
crossover_rate: 0.9
inference_steps: 4
deterministic: False
```

- default_prompt: used if you press Enter at the prompt
- pop_size: number of individuals per generation
- num_generations: maximum generations to run
- mutation_rate: probability per embedding element to mutate
- crossover_proportion: fraction blended during arithmetic crossover (incl. pooled embeddings)
- crossover_rate: chance to crossover vs. cloning
- inference_steps: diffusion steps (Turbo works well with low steps)
- deterministic: toggles seeded generation in the pipeline

## How it works

- Entry: `src/main.py` (`GeneticAlgorithm`)
- Image generation: `evolutionary_prompt_embedding.image_creation.SDXLPromptEmbeddingImageCreator` creates prompt embeddings and renders via Diffusers
- Operators: `evolutionary_prompt_embedding.variation` (pooled arithmetic crossover, uniform Gaussian mutation for standard and pooled embeddings)
- Selection: `evolutionary.evolutionary_selectors.TournamentSelector`
- Display/input: OpenCV window + `pynput` keyboard listener

Saved images are written under `saved_images/<unix_ts>_<sanitized_prompt>/generation<g>_individual_<i>.png`.

## Project structure

```
latent-navigator/
├─ config.yaml
├─ requirements.txt
├─ setup.py
├─ src/
│  ├─ main.py
│  ├─ evolutionary/
│  ├─ evolutionary_imaging/
│  └─ evolutionary_prompt_embedding/
└─ saved_images/
```

## Troubleshooting

- ModuleNotFoundError: evolutionary... → run with `PYTHONPATH=src` or `pip install -e .`
- PyTorch/CUDA install issues → install a build matching your system from https://pytorch.org
- Model downloads/auth → ensure network access; Diffusers fetches models on first run
- No GUI/blank window → OpenCV needs a display; on headless, use a desktop session or Xvfb
- Out of memory (GPU/CPU) → lower `inference_steps`, reduce `pop_size`, close other GPU apps; the code retries pipeline init once

## License

MIT — see `LICENSE`.

## Acknowledgements

- Hugging Face Diffusers
- `stabilityai/sdxl-turbo`

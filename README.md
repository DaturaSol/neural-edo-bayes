# An Empirical Analysis and Optimization of GRU-ODE-Bayes

This repository contains the final project for the course **Notions of Artificial Intelligence (NIA - ENE0082)** at the University of Brasília (UnB).

- **Student:** Gabriel Martins Silveira de Oliveira
- **ID:** 190042656

---

## Abstract

This project presents a deep, practical investigation into the implementation, optimization, and performance of continuous-time models for sporadically-observed time series. 
The primary focus is a critical re-evaluation of the **GRU-ODE-Bayes** model, a state-of-the-art architecture combining Gated Recurrent Units (GRUs) with Neural Ordinary Differential Equations (NODEs).

Through a series of experiments on real-world physiological datasets, this work documents the journey of making the model trainable, from robust data engineering to a fully vectorized implementation using the adjoint method. The investigation reveals two surprising findings:

1.  A CPU-based training pipeline significantly outperforms its GPU counterpart for this specific architecture due to the high overhead of the adjoint-based ODE solver.
  
2.  A simpler baseline model, which omits the complex ODE component entirely, is not only orders of magnitude faster to train but also achieves substantially higher classification accuracy.

These results challenge the practical utility of the GRU-ODE-Bayes architecture for certain classification tasks and highlight a crucial gap between theoretical promise and empirical performance. 
The project also includes a successful application of NODEs to a classic dynamical systems problem (Lotka-Volterra) to demonstrate the power of the concept in its intended domain.

## Project Structure

The repository is organized into the following directories:

```
.
├── src/                  # All Python source code for models, utils, and training pipelines.
├── tex/Template/         # LaTeX source for the accompanying student article.
├── notebooks/            # Jupyter/Colab notebooks for experimentation and visualization.
├── markdown/             # Marp source for the final presentation (in Portuguese).
└── references/           # .bib file containing all references used for this study.
```

### `src/`
This directory contains the core Python source code for the project.
- **`models/`**: Implementations of the GRU-ODE-Bayes model, the baseline model, and the Neural ODE for the Lotka-Volterra experiment.
- **`utils/`**: Helper functions for data preprocessing, chunking, and evaluation.
- **`train.py`**: The main script to run the training and evaluation pipeline.

### `tex/Template/`
This directory contains the LaTeX files for the short student article that summarizes the project's findings, methodology, and conclusions. The main file can be compiled to produce a PDF document.

### `notebooks/`
This folder holds Jupyter/Colab notebooks used for exploratory data analysis, visualization of results (e.g., the Lotka-Volterra phase portraits), and initial model prototyping.

### `markdown/`
This directory contains the Marp markdown source file for the final project presentation, delivered in Portuguese.

### `references/`
This folder contains `references.bib`, a BibTeX file with citations for the key papers and resources used throughout this project.

## Dependency Management with Poetry

This project uses [Poetry](https://python-poetry.org/) for robust dependency management and reproducibility. 
All required packages are defined in the `pyproject.toml` and `poetry.lock` files.

### Core Dependencies
- **Python:** 3.13+
- **PyTorch:** The primary deep learning framework.
- **`torchdiffeq`:** A library for solving ordinary differential equations, used for the Neural ODE implementation with the adjoint method.
- **NumPy, Pandas, Scikit-learn:** For data manipulation and evaluation.
- **Matplotlib:** For plotting and visualization.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DaturaSol/neural-edo-bayes
    cd neural-edo-bayes
    ```

2.  **Install Poetry:**
    If you do not have Poetry installed, follow the [official installation instructions](https://python-poetry.org/docs/#installation). This project was developed using Poetry version `1.2.1` or later.

3.  **Install Project Dependencies:**
    With Poetry installed, run the following command in the root directory of the project. This will create a virtual environment and install all the exact package versions specified in `poetry.lock`.
    ```bash
    poetry install --no-root
    ```

4.  **Activate the Virtual Environment:**
    To run the scripts, you must first activate the virtual environment managed by Poetry.
    ```bash
    poetry env activate
    ```

## Acknowledgments

The implementation of the GRU-ODE-Bayes model was inspired by and adapted from the original work by E. De Brouwer et al. The original repository can be found at:
[https://github.com/edebrouwer/gru_ode_bayes](https://github.com/edebrouwer/gru_ode_bayes)

This project also heavily references the foundational paper "Neural Ordinary Differential Equations" by Chen et al. (2018).

# Interpretability Analysis of Arithmetic In-Context Learning in Large Language Models

This repository will host the code for reproducing the experiments presented in our paper:
> **Interpretability Analysis of Arithmetic In-Context Learning in Large Language Models**
> Gregory Polyakov, Christian Hepting, Carsten Eickhoff, Seyed Ali Bahrainian
> https://aclanthology.org/2025.emnlp-main.92/
> EMNLP 2025 

*We are currently preparing the public release. The repository is being updated*

## Installation

This project requires **Python 3.11**.

Install the main dependencies using `pip`:

```bash
pip install -r requirements.txt
```

To use the **information flow routes** approach, it is also necessary to install the `llm-transparency-tool` directly from its repository.

```bash
git clone https://github.com/facebookresearch/llm-transparency-tool.git
pip install ./llm-transparency-tool
```

## Project Structure

```
.
├── notebooks
│   ├── function_vector_experiments_llama_3.ipynb LLama-3.1-8B FV experiments
│   ├── function_vector_experiments_pythia_12b.ipynb Pythia-12B FV experiments
│   ├── information_flow_routes_experiments_llama_3.ipynb # Llama-3.1-8B Information Flow Routes
│   ├── information_flow_routes_experiments_opt_6_7b.ipynb # OPT-6.7B Information Flow Routes
│   ├── patching_experiments_llama_3.ipynb # Llama-3.1-8B Patching experiments (TBD)
│   ├── patching_experiments_pythia_12b.ipynb # Pythia-12B Patching experiments (TBD)
│   └── partial_sums_experiments_pythia_12b.ipynb # Pythia-12B Partial Sums experiments
├── README.md
├── requirements.txt
└── src
    ├── component.py # FV supporting class
    ├── function_vectors.py # FV supporting functions
    ├── generate_data.py # Data generation scripts (TBD)
    ├── information_flow_routes.py # Information Flow Routes supplementary
    ├── patching.py # Patching supplementary (TBD)
    ├── partial_sums.py # Partial sums supplementary
    └── utils.py # General utility functions
```

## Dataset Generation

All data used in our experiments is available here: **[Google Drive link](https://drive.google.com/drive/folders/1rZDprv4_IjZS_FV1Kx8PMyD2EKiqPHvh?usp=sharing)**

Download the necessary files and put them into the `data/` directory.

*Details on how to generate these datasets will be provided soon.*

## Main Experiments

### Activation Patching

<!-- Our experiments with Activation Patching are presented in:

* `notebooks/patching_experiments_llama_3.ipynb`
* `notebooks/patching_experiments_pythia_12b.ipynb` -->

*To be added soon.*

### Activation Patching with Corrupted Prompts

*To be added soon.*

### Information Flow Routes

Our Information Flow Analysis experiments are presented in:

* `notebooks/information_flow_routes_experiments_llama_3.ipynb`
* `notebooks/information_flow_routes_experiments_opt_6_7b.ipynb`

All helper functions used to generate Information Flow Graphs are located in `src/information_flow_routes.py`.

### Function Vectors

Our experiments with Function Vectors are presented in:

* `notebooks/function_vector_experiments_llama_3.ipynb`
* `notebooks/function_vector_experiments_pythia_12b.ipynb`

All helper functions used to generate Function Vectors are located in `src/function_vectors.py`.

### Partial Sums Representations

Our experiments with probing for partial sums representations are presented in: 

* `notebooks/partial_sums_experiments_pythia_12b.ipynb`

All helper functions used to probe for Partial Sums Representations are located in `src/partial_sums.py`.

## Citation

If you find our paper or code helpful, please cite our paper:
```
@inproceedings{polyakov2025interpretability,
  title={Interpretability Analysis of Arithmetic In-Context Learning in Large Language Models},
  author={Polyakov, Gregory and 
    Hepting, Christian and 
    Eickhoff, Carsten and 
    Bahrainian, Seyed Ali},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={1758--1777},
  year={2025}
}
```
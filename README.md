# Clear2026_Submission_3
This is an anomynised code repo for CLEAR2026 submission #3, entitled: Mechanism learning: reverse causal inference in the presence of multiple unknown confounding through causally weighted Gaussian mixture models

This repository contains the supplementary Python code for our study. It provides:

- A Python package `mechanism_learn` implementing **mechanism learning** and the full pipeline for learning predictors from deconfounded data.
- Jupyter notebooks for the **main experiments** on fully synthetic, semi-synthetic, and real-world ICH (intracranial hemorrhage) detection tasks.
- Additional scripts for **repeated runs**, **ablation / sensitivity analyses** (e.g., varying the number of mixture components, and **visualizations** of the learned GMMs.
- Pre-generated datasets and data-generation scripts for all synthetic and semi-synthetic experiments, as well as processed tabular and feature data for the ICH task.

## Mechanism Learning

A major limitation of machine learning (ML) prediction models is that they recover associational, rather than causal, predictive relationships between variables. In high-stakes automation applications of ML, this is problematic, as the model often learns spurious, non-causal associations. This paper proposes mechanism learning, a simple method which uses causally weighted Gaussian Mixture Models (CW-GMMs) to deconfound observational data such that any appropriate ML model is forced to learn predictive relationships between effects and their causes (reverse causal inference), despite the potential presence of multiple unknown and unmeasured confounding. Effect variables can be very high-dimensional, and the predictive relationship nonlinear, as is common in ML applications. This novel method is widely applicable: the only requirement is the existence of a set of mechanism variables mediating the cause (prediction target) and effect (feature data), which is independent of the (unmeasured) confounding variables. We test our method on fully synthetic, semi-synthetic and real-world data sets, demonstrating that it can discover reliable, unbiased, causal ML predictors, whereas the same ML predictor trained naively using classical supervised learning on the original observational data, is heavily biased by spurious associations.

## Usage instructions

### Requirements

The main experiments were tested with:

- Python 3.9+
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow` - recommend version: 2.10.0
- `matplotlib`
- `grapl-causal` – for graph / ADMG utilities used in the causal weight estimation scripts (where applicable)
- `causalbootstrapping == 0.2.0` – for causal weight estimation and front-door causal bootstrapping

To install the Python dependencies with `pip`, you can use for example:

```bash
pip install numpy pandas scikit-learn matplotlib grapl-causal causalbootstrapping==0.2.0 tensorflow==2.10.0
```

### Installation and usage

To run the experiment code, please also install the (local) mechanism-learn package using the distribution file ``mechanism_learn-2.3.1-py3-none-any.whl`` under the directory ``./code/dist``, using:

```bash
pip install code/dist/mechanism_learn-2.3.1-py3-none-any.whl
```

In Python, to import the mechanism learning algorithms. Using examples are demonstrated in ``./code/main expr`` folder, run: 

```python
import mechanism_learn.pipeline 
```

## Repository structure and remark

```text
./
├── code/
│   ├── mechanism_learn/
│   ├── main expr/
│   ├── additional expr/
│   └── dist/
├── test_data/
│   ├── synthetic_data/
│   │   ├── synthetic_data_gen.py
│   │   ├── syn_classification/
│   │   └── syn_regression/
│   ├── semi_synthetic_data/
│   │   └──  semi_synthetic_data_gen.py
│   └── ICH_data/
│       ├── hemorrhage_diagnosis_ct_clean.csv
│       ├── mediator_embedding.csv
│       ├── ct_clean/
│       │   └── ct_image_selection.py
└──     └── autoencoder/
```

- ``code/``
   - ``/mechanism_learn``: Core algorithm library implementing mechanism learning and CW-GMM methods.
   - ``/main expr``: Jupyter notebooks for reproducing the main experiments in the paper, including:
       - `/synthetic_classification.ipynb`: Main experiment on the synthetic **classification** task.
       - `/synthetic_regression.ipynb` : Main experiment on the synthetic **regression** tasks.
       - `/semi_synthetic_classification.ipynb`: Main experiment on **semi-synthetic** classification task (Background-MNIST).
       - `/ICH_dtection.ipynb`: Main experiment on **ICH (intracranial hemorrhage) detection**. 
   - ``/additional expr``: Other experiments reported in the paper, including:
       - `/visual_gmm_with_diff_K.py`: Visualization of GMM fitting results for different numbers of components.
       - `/metrics_over_K_syn.py`, `metrics_over_K_ICH.py`: Scripts for evaluating metrics across different numbers of mixture components.
       - `/repeated_expr_for_synthetic_classification.py`, `repeated_expr_for_synthetic_regression.py`, `repeated_expr_for_semiSyn_classification.py`, `repeated_expr_for_ICH_detection.py`: Scripts for repeated runs of the four main tasks above, used to compute averaged performance and standard deviations.
       - `/mean_diff_syn_clas.py`, `AME_syn_reg.py`: Analysis script for performance differences on synthetic classification and regression tasks.
       - `/evaluator_utils.py`: Utility functions for evaluation and plotting (common metric computation, etc.).     
- ``/test_data``
  - `/synthetic_data`: Contains all synthetic datasets for synthetic classification and regression tasks. The script `synthetic_data_gen.py` is the source code to generate these synthetic data.
  - `/semi_synthetic_data/`: The script `semi_synthetic_data_gen.py` is the source code to manipulate the MNIST dataset to generate Background-MNIST samples. The generated samples are too large to be included on Github repo.
  - `/ICH_data`: Data and intermediate representations for the ICH (intracranial hemorrhage) detection task.
      - `/hemorrhage_diagnosis_ct_clean.csv`: Cleaned table of CT cases and hemorrhage diagnosis labels.
      - `/mediator_embedding.csv`: mechanism embeddings (features) extracted via an autoencoder from highlighted segmentations.
      - `/ct_clean`: The script `ct_image_selection.py` is the source code for CT image selection and cleaning. The original ICH data is available in the study Hssayeni, M. (2020). Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation (version 1.3.1). PhysioNet. RRID:SCR_007345.
      - `/autoencoder`: Includes model files for the trained autoencoder for mechanism variable embedding and the source code to train the autoencoder.


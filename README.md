![Python](https://img.shields.io/badge/Python-3.10.13-blue)
![PyTorch Lightning](https://img.shields.io/badge/PyTorch--Lightning-792EE5?style=flat&logo=lightning&logoColor=white)
[![Project Page](https://img.shields.io/badge/Project-Page-green?style=flat&logo=github)](https://example.com)

<div align="center">

# Robustness Meets Uncertainty: EV-AT

**Official implementation of the ECCV 2026 paper: [Robustness Meets Uncertainty: Evidential Adversarial Training for Robust Selective Classification](https://github.com/NicolasSournac/EV-AT) by Nicolas Sournac, Ahmed Baha Benjaa and Bertrand Braeckeveldt**

*Multitel Research & Innovation Center, Artificial Intelligence Department, Belgium*

</div>

## Overview

**EV-AT** (short for "Evidential Adversarial Training") is an adversarial training algorithm designed for trustworthy machine learning. The goal is to allow adversarially robust selective classification.

This repository contains the code for our benchmark in robust selective classification, as well as the implementation of our proposed method EV-AT. The benchmark includes multiple adversarial training algorithms, data augmentation strategies, and evaluation metrics to provide a comprehensive comparison of different approaches in this domain.

## Table of Contents
- [⚡ Installation](#⚡-installation)
    - [Using UV](#using-uv)
- [🚀 Getting Started](#rocket-getting-started)
    - [Reproduce our results](#reproduce-our-results)
- [🔧 Extending the Benchmark](#wrench-extending-the-benchmark)
    - [Adding a New Dataset](#adding-a-new-dataset)
    - [Adding a New Model](#adding-a-new-model)
    - [Adding a New Training Pipeline](#adding-a-new-training-pipeline)
- [References](#references)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

##  ⚡ Installation

To setup EV-AT, you must use the provided `pyproject.toml` and setup a virtual environment. For the virtual environment, we recommend using [`uv`](https://github.com/astral-sh/uv).

### Using UV

Simply run the following commands in the root directory of the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
uv sync
```

## :rocket: Getting Started

In order to run properly, the code requires some environment variables to be set by the user. Default values for these variables are provided in the `.env.example` file.

### Reproduce our results

We provide automated scripts to generate and run all benchmark jobs for reproducibility. There are two main workflows: generating all jobs, or running a single job directly.

#### 1️⃣ Generate all benchmark jobs

Use the `generate_benchmark_jobs.sh` script to create all jobs for a **specific model and dataset**. This will generate a subfolder containing all training and evaluation scripts, along with a `run_all.sh` script to launch them sequentially.

```bash
# Usage
bash generate_benchmark_jobs.sh <MODEL> <DATASET> [SEEDS...]
```

**Parameters:**

| Parameter | Options                                |
| --------- | -------------------------------------- |
| MODEL     | `wideresnet_34_10` or `preactresnet18` |
| DATASET   | `cifar10` or `cifar100`                |
| SEEDS     | Optional list of random seeds (default: `0 1 2`) |

**Example:**

```bash
cd jobs/
bash generate_benchmark_jobs.sh wideresnet_34_10 cifar10 0 1 2
```
This will create a folder like:
```
benchmark_wideresnet_34_10_cifar10/
    run_all.sh
    wideresnet_34_10_cifar10_at_basic_s0.sh
    wideresnet_34_10_cifar10_at_basic_s1.sh
    ...
    wideresnet_34_10_cifar10_evat_autoaug_s2.sh
```

To run all jobs sequentially:
```bash
cd benchmark_wideresnet_34_10_cifar10/
sh run_all.sh
```

#### 2️⃣ Run a single job directly

If you want to run a specific configuration, use run_single_job.sh. It automatically generates the correct job script and runs it immediately.

```bash
bash run_single_job.sh <MODEL> <DATASET> <AUG> <PIPE> <SEED>
```
**Parameters:**
| Parameter | Options                                                       |
| --------- | ------------------------------------------------------------- |
| MODEL     | `wideresnet_34_10` or `preactresnet18`                        |
| DATASET   | `cifar10` or `cifar100`                                       |
| AUG       | `basic`, `cutout`, `augmix`, `autoaug`                        |
| PIPE      | `at`, `at_awp`, `trades`, `trades_awp`, `ikl`, `emff`, `evat` |
| SEED      | `0`, `1`, `2`, ...                                           |

**Example:**
```bash
cd jobs/
# Run baseline AT
bash run_single_job.sh wideresnet_34_10 cifar10 basic at 0

# Run TRADES + AWP
bash run_single_job.sh wideresnet_34_10 cifar10 augmix trades_awp 1

# Run EMFF
bash run_single_job.sh wideresnet_34_10 cifar10 cutout emff 2

# Run EV-AT (Our) method
bash run_single_job.sh wideresnet_34_10 cifar10 autoaug evat 0
```

#### 3️⃣ Run a single L2 job
For L2 norm threat model, use the dedicated script run_single_job_l2.sh:

```bash
bash run_single_l2_job.sh <MODEL> <DATASET> <AUG> <PIPE> <SEED>
```
**Parameters:**
| Parameter | Options                                                       |
| --------- | ------------------------------------------------------------- |
| MODEL     | `wideresnet_34_10` or `preactresnet18`                        |
| DATASET   | `cifar10` or `cifar100`                                       |
| AUG       | `basic`, `cutout`, `augmix`, `autoaug`                        |
| PIPE      | `at`, `at_awp`, `trades`, `trades_awp`, `ikl`, `emff`, `evat` |
| SEED      | `0`, `1`, `2`, ...                                           |

**Example:**
```bash
cd jobs/
bash run_single_l2_job.sh preactresnet18 cifar10 cutout ikl 2
```

#### 4️⃣ Export results
After your jobs finish, raw results are stored in the output directories. To convert them into CSV tables:
```bash
python src/export_results.py --root_dir=<output_dir>
```

## :wrench: Extending the Benchmark

The benchmark is designed to be easily extensible. You can add new datasets, models, and training pipelines with minimal effort. This section provides detailed instructions for each type of extension.

### Adding a New Dataset

To add a new dataset, you need to create a new **DataModule class** in `src/ev_at/data/loader.py`.

1️⃣ **Inherit from `LightningDataModule`**:
   ```python
   class MyCustomDatamodule(LightningDataModule):
       def __init__(self, data_path: str | Path, batch_size: int = 32, ...):
           super().__init__()
           # Initialize your parameters
   ```

2️⃣ **Implement required methods**:
   - `prepare_data()`: Download or prepare the dataset (called only once)
   - `setup(stage)`: Load and split the dataset (called on each GPU/process)
   - `train_dataloader()`: Return training DataLoader
   - `val_dataloader()`: Return validation DataLoader
   - `test_dataloader()`: Return test DataLoader

3️⃣ **Define the `num_classes` property**:
   ```python
   @property
   def num_classes(self) -> int:
       return 10  # or your dataset's number of classes
   ```

### Adding a New Model

To add a new model architecture, you need to create a new model file in `src/ev_at/nn/` and corresponding configuration files.

1️⃣ **Create the model file** in `src/ev_at/nn/my_model.py`:
   ```python
   import torch.nn as nn
   
   class MyModel(nn.Module):
       def __init__(self, num_classes: int = 10, ...):
           super().__init__()
           # Define your architecture
           
       def forward(self, x):
           # Define forward pass
           return x
   ```

2️⃣ **Create configuration files** in `configs/model/`:

```yaml
model:
    _target_: ev_at.nn.my_model.MyModel
    num_classes: ${datamodule.num_classes}
    # Add other hyperparameters
```

3️⃣ **Update imports** in `src/ev_at/nn/__init__.py`:
   ```python
   from .my_model import MyModel
   ```

### Adding a New Training Pipeline

To add a new adversarial training algorithm or training strategy, create a new **Lightning Module** in `src/ev_at/pipe/training/`.

#### Required Steps:

1️⃣ **Create the training module** in `src/ev_at/pipe/training/my_method.py`:
   ```python
   from ev_at.core.pipe import TrainingModule
   
   class MyMethodTrainingModule(TrainingModule):
       def __init__(self, model, optimizer, scheduler, ...):
           super().__init__(model, optimizer, scheduler, num_classes=num_classes)
           # Initialize method-specific parameters
   ```

2️⃣ **Implement the `training_step` and `validation_step` methods** (required):
   ```python
   def training_step(self, batch, batch_idx):
       images, labels = batch
       
       # Your training logic here
       # - Generate adversarial examples
       # - Compute loss
       # - Log metrics
       
       loss = ...  # Compute your loss
       self.log("train_loss", loss)
       
       # Update metrics
       logits, preds, _ = self.model(images)
       self.train_metrics.update(preds, labels)
       
       return loss
   ```

```python
def validation_step(self, batch, batch_idx):
    
    # Follow same logic as other training pipelines 
    # to ensure consistent validation

    loss = ... # Compute your validation loss
    self.log("val_loss", loss)
    self.val_metrics.update(preds, labels)
    return loss
```

3️⃣ **Create configuration files** in `configs/`:
```yaml
experiment:
    training_pipe: my_method
```
4️⃣ **Update imports** in `src/ev_at/pipe/training/__init__.py`:
   ```python
   from .my_method import MyMethodTrainingModule
   ```

5️⃣ **Add to `src/train.py`**:
```python
from ev_at.pipe.training import MyMethodTrainingModule
```

```python
if exp_cfg["training_pipe"] == "my_method":
    training_module = MyMethodTrainingModule(
        module, 
        optimizer, 
        scheduler, 
        ...
    )
```

## References
```
@inproceedings{sournac2026evat, 
	author = {Nicolas Sournac and Ahmed Baha Benjaa and Bertrand Braeckeveldt}, 
	title = {Robustness Meets Uncertainty: Evidential Adversarial Training for Robust Selective Classification}, 
	booktitle = {European Conference on Computer Vision (ECCV)},
	year = {2026}
}
```

## Contact
Please contact [sournac@multitel.be](mailto:sournac@multitel.be), [ahmedbaha.benjmaa@outlook.fr](mailto:ahmedbaha.benjmaa@outlook.fr) and [braeckeveldt@multitel.be](mailto:braeckeveldt@multitel.be) 

## Acknowledgements
This work was supported by the Public Service of Wallonia (Economy, Employment and Research) under grant no.~2010235 (ARIAC - Applications and Research for Trusted Artificial Intelligence), within the DigitalWallonia4.ai programme. The research also benefited from computational resources made available on Lucia, the Tier-1 supercomputer of the Walloon Region, infrastructure funded by the Walloon Region under the grant agreement no.~1910247. 
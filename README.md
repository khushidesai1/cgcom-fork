# CGCom
Forked repository of CGCom method. Original source code can be found at: https://github.uconn.edu/how17003/CGCom.

Modified to be an installable Python package for benchmarking purposes.

# Graph Attention Network (GAT) for Cell Communication Analysis

This codebase implements a Graph Attention Network (GAT) model for analyzing cell-to-cell communication in single-cell data, particularly for mouse gastrulation datasets.

## Installation

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/khushidesai/cgcom-fork.git
```

## Directory Structure

Ensure your dataset follows this structure:
```
Dataset/
  MouseGastrulation/
    E1_expression_median_small.csv
    E1_label.csv
    E1_location.csv
    E2_expression_median_small.csv
    E2_label.csv
    E2_location.csv
    E3_expression_median_small.csv
    E3_label.csv
    E3_location.csv
```

Also ensure you have the ligand-receptor knowledge file:
```
Knowledge/
  allr.csv
```

## Usage

### Using as a Python package

You can import the modules and functions directly:

```python
# Import the GAT model
from cgcom.models import GAT

# Import utility functions
import cgcom.utils as utils

# Train the model
from cgcom.scripts.training import train_gat_model
train_gat_model()

# Analyze cell communication
from cgcom.scripts.communication_score import analyze_cell_communication
analyze_cell_communication()
```

### Running the Scripts Directly

1. Training the GAT model:
```bash
python -m cgcom.scripts.training
```

2. Analyzing cell communication:
```bash
python -m cgcom.scripts.communication_score
```

## Package Structure

```
cgcom/
  ├── models/                    # Model implementations
  │   ├── __init__.py           # Exports the GAT model
  │   └── gat.py                # GAT model implementation
  ├── utils/                     # Utility functions
  │   ├── __init__.py           # Exports utility functions
  │   └── data_utils.py         # Data loading and processing utilities
  ├── scripts/                   # Scripts for training and analysis
  │   ├── __init__.py
  │   ├── training.py           # Script for training the GAT model
  │   └── communication_score.py # Script for analyzing cell communication
  └── __init__.py
```

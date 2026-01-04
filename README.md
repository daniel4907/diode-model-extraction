# Compact Model Parameter Extraction Framework

A Python framework for extracting semiconductor device compact model parameters from TCAD simulations.

## Current features:
* Fit diode model (saturation current *I_s* and ideality factor *n*) to I-V data
* Generate synthetic diode I-V curves for testing
* Visualize fitted curves and current errors
* Unit tests validating parameter extraction

## Setup

```bash
conda env create -f environment.yml
conda activate compact-model-extraction
```

Then, run the Jupyter notebook found in ```/examples/diode_extraction.ipynb```

## Project structure
- 'src/models.py' - diode model implementation
- 'src/extraction.py' - parameter extraction
- 'src/visualization.py' - plotting helpers for fits and errors
- 'tests/' - unit tests
- 'examples/diode_extraction.ipynb' - diode I-V extraction demo


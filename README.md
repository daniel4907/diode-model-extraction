# Compact Model Parameter Extraction Framework

A Python framework for extracting semiconductor device compact model parameters from TCAD simulations, featuring interactive visualizations and Streamlit app dashboard.

## Current features:

### Parameter Extraction
* Diode: extract $I_s$, $n$, and $R_s$ from synthetic or real data through a .csv file. Supports multi-temperature analysis to extract $E_g$ and visualize $I_s(T)$
* MOSFET: Level 1 model (Schichman-Hodges) to extract $V_{th}$, $k_n$, and $\lambda$ from transfer and output characteristics
* Generate noisy data using realistic synthesis datsets for testing extraction algorithms
* Automatically generate SPICE-compatible model files from extracted parameters

### Visualization
* Interactive physical states: dynamic cross-section diagrams for both diodes and MOSFETs that respond to sliders
* 3D characteristics surfaces: interactive 3D plots using Plotly to visualize device behavior over voltage and temperature ranges
* Automated plotting: 2D plotting for fits, relative errors, and parameter trends

### GUI
Unified Streamlit dashboard that allows no-code interface that can:
1. Generate synthetic data or upload custom CSVs
2. Configure initial guesses and run extractions
3. Inspect results via interactive 2D/3D plots and physical diagrams
4. Export and download SPICE models

## Setup
1. Clone the repository to your local machine
2. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate compact-model-extraction
```

## Usage
Run the dashboard locally with:
```bash
streamlit run app.py
```

Or, run the Jupyter notebooks found in ```/examples/``` directory.
* `examples/diode_extraction.ipynb`: Diode parameter extraction demo
* `examples/mosfet/extraction.ipynb`: MOSFET parameter extraction demo

## Project structure
- `src/models.py` - diode and MOSFET model implementation
- `src/extraction.py` - parameter extraction logic
- `src/visualization.py` - plotting helpers and interactive device diagrams
- `src/utils.py` - SPICE model generation and data utilities
- `tests/` - unit tests
- `examples/` - demonstration notebooks for model extraction
- `app.py` - Streamlit GUI app


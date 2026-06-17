# Reservoir Analysis Platform

Reservoir Analysis Platform is a Streamlit-based workflow for reservoir data review, multi-well analysis, pore typing, and autonomous carbonate pore throat classification.

## Features

- Well map and multi-well data integration
- Conventional pore typing workflows
- Autonomous carbonate pore throat classification from mercury-injection capillary-pressure data
- Capillary-curve morphology features, including entry, middle, tail, span, and curve-complexity behavior
- Secondary classification and manual merge workflow for classified capillary-pressure curves
- Porosity-permeability, FZI, pore throat radius, and PCA feature-space visualizations

## Project Structure

```text
app.py                         # Streamlit entry point
modules/                       # UI modules and analysis logic
train_model.py                 # Model training script
requirements.txt               # Python dependencies
model_config.json              # Model configuration
data/                          # Local input data, ignored by Git
data_split/                    # Generated train/val/test splits, ignored by Git
models/                        # Local trained model files, ignored by Git
outputs/                       # Runtime outputs, ignored by Git
logs/                          # Runtime logs, ignored by Git
```

## Data Policy

Reservoir data, split datasets, generated outputs, logs, and trained model binaries are intentionally not uploaded to GitHub.

Ignored local folders include:

```text
data/
data_split/
models/*.pkl
outputs/
logs/
```

Place local source datasets in `data/`. Generated train/validation/test files can be recreated with:

```bash
python modules/split_dataset.py
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Autonomous Pore Typing Input

The autonomous carbonate pore typing module expects at least these columns:

```text
CPOR_clean
CKH_clean
PC_STRESS_CORR
SW_STRESS_CORR
```

Optional but recommended columns:

```text
PTR_P
PORE_V_P
Core_ID / SampleID / Plug_ID / ID / ReferenceName
WellName_2 + Plug No
```

Each core or plug identifier should represent one capillary-pressure curve and must contain at least two valid saturation-pressure points.

## Deployment

This app can be deployed as a Streamlit service on a server or in Docker. A typical production setup is:

```text
Streamlit app
Nginx reverse proxy
HTTPS certificate
Mounted data/model directories
```

For private reservoir datasets, keep `data/`, `data_split/`, and trained model files outside GitHub and copy or mount them directly on the deployment server.

## Notes

- `data_split/` is generated data and should not be committed.
- Excel temporary lock files such as `~$test.xlsx` are ignored.
- If model files are required in production, place them under `models/` on the server.

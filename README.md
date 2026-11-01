# Human activity recognition with a hidden Markov model

An iPhone activity log for four states: still, standing, walking, and jumping. Sensor Logger recordings are windowed, turned into time- and frequency-domain features, and decoded with a discrete hidden Markov model in `hmm.ipynb`.

## Setup

The project uses Python 3.14 and [pipenv](https://pipenv.pypa.io/). From the repository root:

```bash
pipenv install
```

That installs NumPy, pandas, scikit-learn, SciPy, Matplotlib, seaborn, ipykernel, and JupyterLab from the `Pipfile`.

Register that environment as a Jupyter kernel so JupyterLab can open `hmm.ipynb` with the packages above. The name matches the notebook's kernelspec:

```bash
pipenv run python -m ipykernel install --user --name=ml-tech2-formative-2 --display-name="ml-tech2-formative-2"
```

## Start JupyterLab

The `Pipfile` script `notebook` starts JupyterLab without opening a browser:

```bash
pipenv run notebook
```

## Preprocessing

Sensor Logger zip exports belong in `data/raw/`. Each zip is named with the activity first (`still-…`, `standing-…`, `walking-…`, `jumping-…`) and contains `Accelerometer.csv`, `AccelerometerUncalibrated.csv`, and `Gyroscope.csv`.

```bash
pipenv run python scripts/preprocess.py
```

The script merges the sensors on `seconds_elapsed` and writes non-overlapping 5-second windows (500 rows at 100 Hz) to `data/clean/`.

## Feature extraction

Run this after preprocessing. It reads every CSV in `data/clean/` and writes `data/extracted_features.csv`.

```bash
pipenv run python scripts/feature_extraction.py
```

`data/clean/` and `data/extracted_features.csv` are already in the repository, so the notebook can be tested without regenerating them.

## Test the notebook

1. Start JupyterLab with `pipenv run notebook`.
2. Open `hmm.ipynb` from the repository root. The notebook loads `data/clean/` and `data/extracted_features.csv` with paths relative to that root.
3. Select the `ml-tech2-formative-2` kernel registered above. `ipykernel` is installed by `pipenv install`.
4. Run All.

A successful run prints:

- `Generated 9 training sequences.`
- `Generated 4 validation sequences.`
- Baum–Welch convergence at iteration 66, when the log-likelihood gain falls below `1e-4`.
- Unseen-data accuracy of 100%, with sensitivity and specificity of 100% for still (10 windows), standing (10), walking (9), and jumping (9).

# Capstone 2025 – xG Model (Goal Probability)

This project implements an **expected goals (xG)** model using StatsBomb public data (events in JSON) to predict the probability of scoring for each shot.

In modern football, xG is used by analysts, coaches, and scouting departments to evaluate the quality of chances created and conceded, independently of the actual outcome of each shot. Instead of looking only at goals scored/conceded, xG measures how likely a typical shot is to result in a goal, given its **distance**, **angle**, **game situation**, **body part**, etc.  
The goal of this project is to build a reproducible xG model trained on StatsBomb Open Data and expose it via a simple web service (`/predict`) that can be used by other applications (dashboards, notebooks, etc.).

- Dataset: [StatsBomb Open Data](https://github.com/statsbomb/open-data) (events in JSON)
- Problem: given a shot event (`Shot`), predict:
  - `is_goal` (0/1) and
  - `prob_goal` (calibrated goal probability)
- Models:
  - Logistic Regression (baseline)
  - RandomForestClassifier
  - (Optional) XGBoost / LightGBM / CatBoost, if you install the libs
- Metrics:
  - Log Loss
  - ROC AUC
  - Brier score (calibração)
- Web service:
  - endpoint `POST /predict`
  - receives a JSON with the shot event (StatsBomb format or simplified)
  - returns `prob_goal`

---

## 1. Prepare StatsBomb data

1. Clone the StatsBomb open data repository **on your own machine**:

   ```bash
   cd cohorts/2025/capstone
   git clone https://github.com/statsbomb/open-data.git data/statsbomb-open-data
   ```

2. The event files will be in:

   ```bash
   data/statsbomb-open-data/data/events
   ```

   The training script expects this path by default. You can change it with `--events-dir`.

---

## 2. Python environment

It is recommended to use a dedicated virtual environment (e.g. `venv` or conda) for reproducibility.

### 2.1 Create and activate virtual environment (example with `venv`)

```bash
cd cohorts/2025/capstone

# create virtual environment
python -m venv .venv

# activate (Linux/macOS)
source .venv/bin/activate

# or on Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```

### 2.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies:

- `pandas`, `numpy`
- `scikit-learn`
- `joblib`
- `Flask`
- `matplotlib` (for EDA in the notebook)

If you want to use gradient boosting models:

- `xgboost`, `lightgbm`, `catboost` (optional; the code only uses them if you add them).

To work with the notebook:

- also install `notebook` or `jupyterlab` (if you don’t have them yet):

  ```bash
  pip install notebook
  ```

---

## 3. Notebook: data prep, EDA and model selection

The `notebook.ipynb` file contains:

- Loading shot data from StatsBomb event files (`load_shots_dataframe`);
- Basic data cleaning (removing shots without location, minute, etc.);
- EDA:
  - distribution of distance, angle, minute;
  - goal vs non‑goal proportion;
- Model comparison (Logistic Regression vs RandomForest) with `cross_val_score` (ROC AUC);
- Feature importance analysis using:
  - Logistic Regression coefficients (after `get_dummies`);
  - simple hyperparameter tuning for RandomForest with `GridSearchCV`.

To run the notebook:

```bash
cd cohorts/2025/capstone
jupyter notebook notebook.ipynb
```

---

## 4. Train models

Train Logistic Regression (baseline) with Platt (sigmoid) calibration:

```bash
python train.py logreg \
    --events-dir data/statsbomb-open-data/data/events \
    --calibration sigmoid
```

Train Random Forest without calibration:

```bash
python train.py random_forest \
    --events-dir data/statsbomb-open-data/data/events \
    --calibration none
```

Main arguments for `train.py`:

- `model`: `logreg` or `random_forest`
- `--events-dir`: folder with StatsBomb event `.json` files
- `--calibration`: `none`, `sigmoid` (Platt) or `isotonic`

The script:

- reads all events,
- keeps only shots (`Shot`),
- builds features:
  - distance to goal
  - shot angle
  - minute / second
  - pressure (`under_pressure`)
  - body part (`body_part`)
  - shot type (`shot_type`)
  - play pattern (`play_pattern` – open play, free‑kick, corner, etc.)
- trains the chosen model,
- evaluates on a test set (stratified split) using:
  - LogLoss
  - ROC AUC
  - Brier score
- saves the pipeline (preprocessing + model + calibration) to:

```text
models/xg_<model_name>[_<calibration>].joblib
```

The exact path is printed at the end of training.

---

## 5. Web service `/predict`

The Flask service is in `predict.py`.

1. Make sure you have a trained model (for example, the calibrated logistic regression).
2. Export the model path to the `XG_MODEL_FILE` environment variable (otherwise the default is used):

```bash
export XG_MODEL_FILE=models/xg_logreg_sigmoid.joblib
```

3. Start the service:

```bash
python predict.py
```

By default it runs at `http://0.0.0.0:9696`.

---

## 6. JSON format for `/predict`

The endpoint expects a JSON representing **a shot event** (in StatsBomb format or simplified). Relevant fields:

```json
{
  "minute": 55,
  "second": 12,
  "location": [102.3, 38.7],
  "under_pressure": true,
  "play_pattern": {"name": "From Free Kick"},
  "shot": {
    "outcome": {"name": "Off T"},
    "body_part": {"name": "Right Foot"},
    "type": {"name": "Open Play"},
    "first_time": false
  }
}
```

The features used during training are extracted from this JSON. The response has the format:

```json
{
  "prob_goal": 0.13,
  "features": {
    "distance": 18.5,
    "angle": 0.9,
    "minute": 55,
    "second": 12,
    "under_pressure": 1,
    "first_time": 0,
    "play_pattern": "From Free Kick",
    "body_part": "Right Foot",
    "shot_type": "Open Play"
  }
}
```

---

## 7. Dockerfile and running with Docker

The `Dockerfile` in this project lets you run the Flask service inside a container.

### Build image

After training and saving the model (e.g. `models/xg_logreg_sigmoid.joblib`):

```bash
cd cohorts/2025/capstone
docker build -t xg-service .
```

### Run container

Assuming the model is in `models/xg_logreg_sigmoid.joblib` on your machine:

```bash
docker run -it --rm \
  -p 9696:9696 \
  -e XG_MODEL_FILE=/app/models/xg_logreg_sigmoid.joblib \
  -v $(pwd)/models:/app/models \
  xg-service
```

Then you can call `POST http://localhost:9696/predict` as described above.

---

## 8. Deployment

This section describes how you could deploy to the cloud. You can choose any provider; below is an example using **Google Cloud Run** with the existing `Dockerfile`.

### 8.1 Local build and test (recap)

```bash
cd cohorts/2025/capstone
docker build -t xg-service .

# train the model on your machine and make sure you have models/xg_logreg_sigmoid.joblib
python train.py logreg --events-dir data/statsbomb-open-data/data/events --calibration sigmoid

docker run -it --rm \
  -p 9696:9696 \
  -e XG_MODEL_FILE=/app/models/xg_logreg_sigmoid.joblib \
  -v $(pwd)/models:/app/models \
  xg-service
```

### 8.2 Example: deployment on Google Cloud Run (container)

1. Install the gcloud CLI and log in:

   ```bash
   gcloud auth login
   gcloud config set project <O_TEUS_PROJECT_ID>
   ```

2. Build the image and push it to Artifact Registry/Container Registry:

   ```bash
   cd cohorts/2025/capstone
   gcloud builds submit --tag gcr.io/<O_TEUS_PROJECT_ID>/xg-service .
   ```

3. Deploy to Cloud Run (exposing port 9696):

   ```bash
   gcloud run deploy xg-service \
     --image gcr.io/<O_TEUS_PROJECT_ID>/xg-service \
     --platform managed \
     --region europe-west1 \
     --allow-unauthenticated \
     --set-env-vars XG_MODEL_FILE=/app/models/xg_logreg_sigmoid.joblib
   ```

4. After deployment, Cloud Run returns a public URL, for example:

   ```text
   https://xg-service-abc123-ew.a.run.app
   ```

   You can then send `POST https://xg-service-abc123-ew.a.run.app/predict` with a shot JSON.

For the capstone, you should include here:

- **The service URL** where you deployed (Cloud Run, EC2, Render, etc.) _or_
- **A link to a video/screencast** showing:
  - the service running (locally or in the cloud), and
  - a `/predict` request (e.g. via `curl`, `Postman`, or a simple front‑end).

---

## 9. Possible extensions

- Use StatsBomb `statsbomb_xg` as a baseline and compare models.
- Add assist features (pass type, height, cross) using `key_pass_id`.
- Use XGBoost/LightGBM/CatBoost to compare performance with Logistic Regression.
- Calibrate by team/competition or by game period.

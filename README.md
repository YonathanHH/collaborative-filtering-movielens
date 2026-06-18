# 🎬 MovieLens 1M — Collaborative Filtering

Collaborative filtering recommender system built on the [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) dataset.

## Models
| Model | Method | Notes |
|-------|--------|-------|
| **BaselineOnly** | ALS bias optimisation | Predicts using `μ + b_u + b_i`. ALS is the *solver* for the biases, not full matrix factorisation. |
| **SVD Default** | Matrix factorisation | Default hyperparameters |
| **SVD Tuned** | Matrix factorisation | Best params found via `GridSearchCV` |

## Repository Structure
```
movielens_cf/
├── app.py                          # Streamlit app
├── requirements.txt
├── README.md
├── data/
│   ├── ratings.csv                 # user_id, movie_id, rating, timestamp
│   ├── movies.csv                  # movie_id, title, genres
│   └── users.csv                   # user_id, gender, age, occupation, zip_code
├── model/
│   ├── svd_model.pkl               # Best tuned SVD (joblib)
│   ├── baseline_als.pkl            # BaselineOnly (ALS bias)
│   ├── item_sim_df.pkl             # Cosine similarity matrix between movies
│   ├── movie_lookup.pkl            # Dict: movie_id → title
│   ├── best_svd_params.pkl         # Best GridSearchCV params
│   └── model_comparison.csv        # RMSE/MAE comparison table
└── collaborative_filtering_movielens.ipynb
```

## Quickstart

### 1. Prepare Data
Download [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and convert `.dat` → `.csv`:

```python
import pandas as pd

ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None,
                      names=['user_id','movie_id','rating','timestamp'],
                      engine='python')
movies  = pd.read_csv('ml-1m/movies.dat',  sep='::', header=None,
                      names=['movie_id','title','genres'],
                      engine='python', encoding='latin-1')
users   = pd.read_csv('ml-1m/users.dat',   sep='::', header=None,
                      names=['user_id','gender','age','occupation','zip_code'],
                      engine='python', encoding='latin-1')

ratings.to_csv('data/ratings.csv', index=False)
movies.to_csv('data/movies.csv',   index=False)
users.to_csv('data/users.csv',     index=False)
```

### 2. Run the Notebook
Open and run all cells in `notebook/collaborative_filtering_movielens.ipynb`.  
This trains all models and saves artefacts to `model/`.

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

## Key Concept — Why is it called ALS in the notebook?

`BaselineOnly(bsl_options={'method': 'als'})` uses **ALS as the optimiser** to estimate
user and item biases inside a simple bias model — not as a full matrix factorisation
recommender. The training output says *"Estimating biases using als..."* which refers to
the *solver*, not the model type.

## Tech Stack
- [scikit-surprise](https://surprise.readthedocs.io/) — collaborative filtering
- [scikit-learn](https://scikit-learn.org/) — cosine similarity
- [Streamlit](https://streamlit.io/) — web app
- [joblib](https://joblib.readthedocs.io/) — model persistence

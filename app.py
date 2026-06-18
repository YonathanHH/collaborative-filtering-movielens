import streamlit as st
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MovieLens Recommender",
    page_icon="🎬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts  (cached so they only load once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = "model"
    item_sim_df   = joblib.load(os.path.join(base, "item_sim_df.pkl"))
    movie_lookup  = joblib.load(os.path.join(base, "movie_lookup.pkl"))
    best_params   = joblib.load(os.path.join(base, "best_svd_params.pkl"))
    comparison_df = pd.read_csv(os.path.join(base, "model_comparison.csv"))
    return item_sim_df, movie_lookup, best_params, comparison_df

@st.cache_data
def load_movies():
    return pd.read_csv("data/movies.csv")

item_sim_df, movie_lookup, best_params, comparison_df = load_models()
movies_df = load_movies()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Grouplens-logo.png/320px-Grouplens-logo.png")
    st.markdown("## 🎬 MovieLens Recommender")
    st.markdown("Collaborative filtering on **1 million** ratings from **6,040 users** across **3,900+ movies**.")
    st.divider()

    st.markdown("### About the Models")
    st.markdown("""
**BaselineOnly (ALS bias)**  
Predicts `r̂ = μ + b_u + b_i` using user/item biases estimated with ALS optimisation.  
*Simple but fast.*

**SVD (Tuned)**  
Matrix factorisation — decomposes the rating matrix into latent user & item factors.  
*More powerful, used for similarity search.*
    """)

    st.divider()
    st.caption("Built with scikit-surprise · Streamlit")
    st.caption("Dataset: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🎬 MovieLens Collaborative Filtering")
st.markdown("Find movies similar to one you love — powered by **SVD latent-factor similarity**.")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Find Similar Movies", "📊 Model Comparison"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Movie Similarity Search
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Search for a movie to get recommendations")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Autocomplete-style select box from all movie titles
        all_titles = sorted(movies_df["title"].dropna().unique().tolist())
        selected_title = st.selectbox(
            "🎥 Select or type a movie title:",
            options=all_titles,
            index=all_titles.index("Toy Story (1995)") if "Toy Story (1995)" in all_titles else 0,
            help="Start typing to filter the list"
        )

    with col2:
        top_n = st.slider("Top N results", min_value=5, max_value=20, value=10)

    search_btn = st.button("🔍 Find Similar Movies", type="primary", use_container_width=True)

    if search_btn or selected_title:
        # Look up movie_id from title
        match = movies_df[movies_df["title"] == selected_title]

        if match.empty:
            st.error(f"Movie not found: {selected_title}")
        else:
            movie_id = match.iloc[0]["movie_id"]
            genre    = match.iloc[0]["genres"] if "genres" in match.columns else "N/A"

            # Check if movie is in similarity matrix
            if movie_id not in item_sim_df.index:
                st.warning(
                    "⚠️ This movie does not have enough ratings to appear in the "
                    "similarity matrix. Please try another title."
                )
            else:
                # Get similarity scores
                scores = item_sim_df[movie_id].sort_values(ascending=False)
                scores = scores[scores.index != movie_id].head(top_n)

                results = pd.DataFrame({
                    "movie_id"  : scores.index,
                    "similarity": scores.values
                })
                results["title"]  = results["movie_id"].map(movie_lookup)
                results["genres"] = results["movie_id"].map(
                    movies_df.set_index("movie_id")["genres"].to_dict()
                ) if "genres" in movies_df.columns else "N/A"
                results = results[["title","genres","similarity"]].reset_index(drop=True)
                results.index += 1
                results["similarity"] = results["similarity"].round(4)

                # Display selected movie info
                st.markdown(f"### 🎯 You selected: **{selected_title}**")
                col_a, col_b = st.columns(2)
                col_a.metric("Movie ID", movie_id)
                col_b.metric("Genre", genre if genre else "N/A")

                st.markdown(f"### 🎬 Top {top_n} Similar Movies")
                st.dataframe(
                    results,
                    use_container_width=True,
                    column_config={
                        "title"     : st.column_config.TextColumn("Movie Title", width="large"),
                        "genres"    : st.column_config.TextColumn("Genres", width="medium"),
                        "similarity": st.column_config.ProgressColumn(
                            "Cosine Similarity",
                            min_value=0, max_value=1, format="%.4f"
                        )
                    }
                )

                # Similarity bar chart
                st.markdown("#### Similarity Scores")
                chart_df = results.set_index("title")["similarity"].head(10)
                st.bar_chart(chart_df)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Model Performance Comparison")
    st.markdown(
        "All models were evaluated on the MovieLens 1M dataset "
        "using **5-fold cross-validation** (RMSE & MAE). "
        "Lower values = better performance."
    )

    col1, col2, col3 = st.columns(3)

    for col, row in zip([col1, col2, col3], comparison_df.itertuples()):
        with col:
            st.markdown(f"#### {row.Model}")
            st.metric("RMSE", f"{row.RMSE:.4f}")
            st.metric("MAE",  f"{row.MAE:.4f}")

    st.divider()

    # Comparison table
    st.dataframe(
        comparison_df.style.highlight_min(subset=["RMSE","MAE"], color="#d4edda"),
        use_container_width=True
    )

    st.divider()

    # Best params
    st.markdown("#### ⚙️ Best SVD Hyperparameters (from GridSearchCV)")
    params_df = pd.DataFrame(best_params.items(), columns=["Parameter","Value"])
    st.dataframe(params_df, use_container_width=True)

    st.info(
        "💡 **Why SVD outperforms BaselineOnly?**  \n"
        "BaselineOnly only learns a single bias number per user and per movie. "
        "SVD learns rich latent vectors (hidden features) for every user and movie, "
        "capturing complex taste patterns that biases alone cannot express."
    )

    # Model explanation expanders
    st.divider()
    st.markdown("#### 📖 Model Explanations")

    with st.expander("BaselineOnly (ALS bias optimisation)"):
        st.markdown("""
**What it does:**  
Predicts ratings using the formula:

```
r̂_ui = μ + b_u + b_i
```

Where:
- **μ** = global mean rating across all users and movies  
- **b_u** = user bias (e.g. a harsh critic always rates 1 star lower than average)  
- **b_i** = item bias (e.g. a blockbuster movie consistently rates 0.5 stars higher)  

**What ALS means here:**  
The biases (b_u and b_i) are estimated using **Alternating Least Squares**.  
It alternates between fixing b_u and solving for b_i, then vice versa, until convergence.  
This is why training prints *"Estimating biases using als..."*

> ⚠️ This is NOT the same as full ALS matrix factorisation (Spark ALS).  
> It is a bias-only baseline with an ALS solver inside.
        """)

    with st.expander("SVD (Singular Value Decomposition)"):
        st.markdown("""
**What it does:**  
SVD decomposes the rating matrix **R** into latent user and item factor matrices:

```
R ≈ U · Σ · Vᵀ
```

- **U** = user latent factor matrix  
- **Σ** = singular values  
- **V** = item latent factor matrix  

Each user and movie gets a vector of hidden features (e.g. "action-loving", "prefers drama").  
Ratings are predicted by the dot product of user and item vectors.

**Why it works better:**  
It captures rich patterns that biases alone cannot — e.g. "User A likes action movies made in the 90s".

**Hyperparameters tuned:**  
- `n_factors` — number of latent dimensions  
- `n_epochs` — training iterations  
- `lr_all` — learning rate  
- `reg_all` — regularisation to prevent overfitting
        """)

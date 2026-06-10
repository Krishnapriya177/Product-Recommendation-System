# 🛒 Product Recommendation System

A full end-to-end machine learning project that builds a **cluster-based product recommendation system** on Amazon product ratings data (~7.8 million rows). The project covers exploratory data analysis, unsupervised user/product clustering with three models, a Bayesian-scoring recommendation engine, and a live Streamlit web application.

---

## 📁 Project Structure

```
product-recommendation-system/
│
├── ratings.csv                            # Raw dataset — 7.8M user-product ratings (no header)
│
├── 1_EDA_and_Visualization.ipynb          # Notebook 1: Data exploration & visualisation
├── 2_Modelling_and_Recommendation.ipynb   # Notebook 2: Feature engineering, clustering, recommendations
│
├── 3_Deployment/
│   ├── app.py                             # Streamlit web application  (source file: "combo")
│   ├── user_features.pkl                  # Serialised user feature + cluster DataFrame
│   ├── product_features.pkl               # Serialised product feature + cluster DataFrame
│   ├── cluster_product_scores.pkl         # Pre-aggregated cluster × product scores
│   └── rating.bin                         # Serialised ratings binary
│
└── requirements.txt
```

> **Note:** The deployment archive contains binary pickle files for all pre-computed artefacts, enabling the Streamlit app to serve recommendations without re-running the notebooks at runtime.

---

## 📊 Dataset — `ratings.csv`

The CSV has **no header row**. Columns are assigned in code:

| Column      | Type    | Description                                      |
|-------------|---------|--------------------------------------------------|
| `userId`    | string  | Unique identifier for each user                  |
| `productId` | int64   | Unique identifier for each product               |
| `Rating`    | float64 | Rating given by the user (scale: 1.0 – 5.0)     |
| `timestamp` | int64   | Unix epoch timestamp *(dropped during analysis)* |

**Dataset scale:**
- ~7,824,481 total ratings
- ~4.2 million unique users
- ~476,000 unique products

---

## 📒 Notebooks

### Notebook 1 — EDA & Visualisation (`1_EDA_and_Visualization.ipynb`)

A thorough data profiling and visual analysis with 19 sections:

| Section | What it covers |
|---------|----------------|
| 1–4 | Library imports, data loading, column renaming, null/duplicate checks |
| 5 | Unique user & product counts, total rating count |
| 6 | Rating distribution — bar chart, pie chart, KDE density plot |
| 7 | Top 10 most-rated products & top 10 most-active users |
| 8 | Average product rating with a minimum-10-ratings filter (avoids cold-start bias) |
| 9 | Histograms of ratings-per-user and ratings-per-product distributions |
| 10 | Outlier detection via IQR method + boxplots + capping treatment |
| 11 | Scatter plot: average rating vs. rating count |
| 12 | Long-tail distribution of product popularity |
| 13 | User rater-type classification (Harsh / Neutral / Generous) with behaviour scatter |
| 14 | Memory-safe pivot table (top 500 users × top 500 products), sparsity stats, heatmap |
| 15 | Sparse matrix (CSR format) construction and full-dataset sparsity calculation |
| 16 | User-product interaction heatmap from sparse matrix sample |
| 17 | Product correlation matrix and heatmap (15 × 15 sample) |
| 18 | Pairplot on 500-row sample |
| 19 | Final insights summary (10 key findings) |

---

### Notebook 2 — Modelling & Recommendation (`2_Modelling_and_Recommendation.ipynb`)

End-to-end ML pipeline in 12 sections:

#### Feature Engineering
- **User feature matrix** (per-user stats): `avg_rating`, `rating_count`, `rating_std`, `min_rating`, `max_rating`, `rating_range`, per-star percentage features (`pct_rating_1` … `pct_rating_5`), and preference profile features derived from products rated ≥ 4
- **Product feature matrix** (per-product stats): same statistical aggregations over all received ratings
- Scaling with `StandardScaler`, then PCA (2 components) for visualisation

#### Clustering Models

| Model | Approach | Scalability technique |
|-------|----------|-----------------------|
| **K-Means** | Elbow method + Silhouette analysis to find optimal K | `MiniBatchKMeans` + 20k-row subsampling |
| **Agglomerative (Ward)** | Hierarchical clustering with dendrogram | Sample-and-propagate via `NearestCentroid` |
| **DBSCAN** | Density-based; noise users get label `-1` | K-distance plot for `eps` tuning; KNN label propagation |

#### Comparative Evaluation

All three models are scored on:
- **Silhouette Score** ↑ (cohesion + separation, range −1 to +1)
- **Davies-Bouldin Index** ↓ (lower = better separated clusters)
- **Calinski-Harabasz Score** ↑ (between-cluster vs within-cluster variance ratio)

The best-scoring model is automatically selected for the recommendation engine.

#### Recommendation Engine

**User-based (Cluster-Enhanced Collaborative Filtering):**
1. Look up the target user's cluster
2. Collect all ratings from cluster-mates
3. Exclude products the user has already rated
4. Score candidates with a **Bayesian average** (`score = (sum + global_mean × c) / (count + c)`, `c = 10`)
5. Return Top-N by score; noise users (`dbscan_cluster = -1`) fall back to global popularity

**Item-based:**
1. Look up the target product's cluster
2. Return cluster-mates sorted by average rating

**Evaluation:** Leave-One-Out Hit Rate @ 10 on 200 sampled users (hides each user's highest-rated product, checks if it appears in Top-10 recommendations).

---

## 🌐 Streamlit Application (`app.py` / `combo`)

A two-mode interactive dashboard:

| Mode | Input | Output |
|------|-------|--------|
| **👤 User-Based** | User ID (text input) | Top-N products recommended from the user's cluster, ranked by popularity × rating |
| **📦 Product-Based** | Product ID (text input) | Top-N similar products from the same product cluster, ranked by avg rating |

**UI features:**
- Sidebar mode selector and "How it works" explainer
- Adjustable Top-N slider (5 – 20)
- Live metric cards: total users/products, cluster count, selected cluster
- Dynamic sidebar showing the selected user's/product's avg rating, total ratings, and cluster
- Fully memory-safe — loads pre-computed pickle files; no runtime DataFrame scanning

### Running the App Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Rename the app entry-point (if still named 'combo')
mv combo app.py

# 3. Place all pickle files in the same directory as app.py:
#    user_features.pkl
#    product_features.pkl
#    cluster_product_scores.pkl

# 4. Launch
streamlit run app.py
```

The app will open at `http://localhost:8501` by default.

---

## ⚙️ Setup & Installation

```bash
# Clone or download the project
git clone <your-repo-url>
cd product-recommendation-system

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
```

**Python version:** 3.9 or higher recommended.

---

## 🔄 End-to-End Pipeline

```
ratings.csv (7.8M rows, no header)
        │
        ▼
1. EDA & Visualisation
   ├── Clean & profile data
   ├── Rating distribution analysis
   ├── User behaviour segmentation (Harsh / Neutral / Generous)
   ├── Pivot table + sparse matrix
   └── Outlier detection & capping
        │
        ▼
2. Feature Engineering
   ├── User feature matrix (14 features per user)
   ├── Product feature matrix (7 features per product)
   └── StandardScaler → PCA 2D
        │
        ▼
3. Clustering
   ├── K-Means (MiniBatch, Elbow + Silhouette)
   ├── Agglomerative Ward (sample + centroid propagation)
   └── DBSCAN (K-distance plot + KNN propagation)
        │
        ▼
4. Model Selection
   └── Best model by Silhouette / Davies-Bouldin / Calinski-Harabasz
        │
        ▼
5. Recommendation Engine
   ├── User-based: cluster-mate Bayesian scoring
   ├── Item-based: product cluster similarity
   └── Leave-One-Out Hit Rate @ 10 evaluation
        │
        ▼
6. Deployment
   └── Streamlit dashboard (app.py + pre-computed pickles)
```

---

## 📈 Model Evaluation Notes

The Leave-One-Out Hit Rate @ 10 reported **0%** on the initial run. Key reasons and suggested fixes:

| Issue | Suggested Fix |
|-------|--------------|
| High Bayesian regularisation (`c_reg=10`) compresses all scores toward the global mean | Reduce `c_reg` to 1–5 |
| K=11 clusters too coarse for 4.2M users | Increase K or use fine-grained sub-clustering |
| LOO hides niche items rarely seen by cluster-mates | Evaluate on randomly hidden items, not always the highest-rated |
| Extreme data sparsity (7.8M ratings / 476K products) | Try matrix factorisation (SVD / NMF) as a hybrid |

---

## 🛠️ Technologies

| Category | Libraries / Tools |
|----------|-------------------|
| Data manipulation | `pandas`, `numpy` |
| Sparse matrices | `scipy` (`csr_matrix`, `linkage`) |
| Visualisation | `matplotlib`, `seaborn` |
| ML — clustering | `scikit-learn` (KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN) |
| ML — preprocessing | `scikit-learn` (StandardScaler, PCA, TruncatedSVD, NearestNeighbors, NearestCentroid, KNeighborsClassifier) |
| ML — evaluation | `scikit-learn` (silhouette_score, davies_bouldin_score, calinski_harabasz_score) |
| Web application | `streamlit` |
| Serialisation | `pickle` (built-in) |
| Notebooks | `jupyter` |

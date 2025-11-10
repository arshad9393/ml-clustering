import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Wholesale customers data_clustering.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SPEND_COLS = ["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]
df = pd.read_csv(DATA_PATH)
X = df[SPEND_COLS].copy().fillna(0)
X_log = np.log1p(X)
scaler = StandardScaler().fit(X_log)
X_scaled = scaler.transform(X_log)

best_k = None
best_sil = -1
results = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    results.append((k, km.inertia_, sil))
    if sil > best_sil:
        best_sil = sil
        best_k = k

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(X_scaled)
gmm = GaussianMixture(n_components=best_k, random_state=42).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=best_k).fit(X_scaled)

def _save(obj, name):
    tmp = os.path.join(MODELS_DIR, f"_{name}.tmp")
    out = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(obj, tmp)
    os.replace(tmp, out)

_save(scaler, "scaler")
_save(kmeans, "kmeans")
_save(gmm, "gmm")
_save(agg, "agg")
centers_scaled = kmeans.cluster_centers_
centers_log = scaler.inverse_transform(centers_scaled)
centers_orig = np.expm1(centers_log)
centroid_df = pd.DataFrame(centers_orig, columns=SPEND_COLS)
centroid_df.to_csv(os.path.join(MODELS_DIR, f"centroids_k{best_k}.csv"), index=False)

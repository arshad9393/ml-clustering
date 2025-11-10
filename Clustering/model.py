import os, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Wholesale customers data_clustering.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

SPEND_COLS = ["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]
df = pd.read_csv(DATA_PATH)
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans.pkl"))
gmm = joblib.load(os.path.join(MODELS_DIR, "gmm.pkl"))
agg = joblib.load(os.path.join(MODELS_DIR, "agg.pkl"))

X = df[SPEND_COLS].copy().fillna(0)
X_log = np.log1p(X)
X_scaled = scaler.transform(X_log)

k_labels = kmeans.predict(X_scaled)
g_labels = gmm.predict(X_scaled)
try:
    from sklearn.cluster import AgglomerativeClustering
    a_labels = agg.labels_
except Exception:
    a_labels = None

df_out = df.copy()
df_out["kmeans_cluster"] = k_labels
df_out["gmm_cluster"] = g_labels
if a_labels is not None:
    df_out["agg_cluster"] = a_labels
out_path = os.path.join(MODELS_DIR, "clustered_dataset.csv")
df_out.to_csv(out_path, index=False)
print("Saved clustered dataset to", out_path)

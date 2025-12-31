# 05_clustering_analysis.py
# Clustering Pola Penjualan Tiket

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8')

# PATH SETUP

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "penjualan_tiket_clean.csv"

print("DATA_PATH :", DATA_PATH)

# LOAD DATA

df = pd.read_csv(DATA_PATH)

print("\n=== INFO DATA ===")
print(df.info())

# SELEKSI FITUR UNTUK CLUSTERING
# Fokus ke variabel numerik yang relevan

features = [
    'jumlah',
    'total',
    'harga_per_tiket',
    'is_weekend',
    'jam'
]

X = df[features]

print("\n=== FITUR CLUSTERING ===")
print(X.head())

# SCALING DATA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ELBOW METHOD

inertia = []

K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('Inertia')
plt.tight_layout()
plt.show()

# K-MEANS CLUSTERING
# k dipilih berdasarkan elbow (umumnya 3)

k_optimal = 3

kmeans = KMeans(
    n_clusters=k_optimal,
    random_state=42,
    n_init=10
)

df['cluster'] = kmeans.fit_predict(X_scaled)

print("\n=== JUMLAH DATA PER CLUSTER ===")
print(df['cluster'].value_counts())

# ANALISIS KARAKTERISTIK CLUSTER

cluster_summary = (
    df.groupby('cluster')[features]
    .mean()
    .round(2)
)

print("\n=== RATA-RATA SETIAP CLUSTER ===")
print(cluster_summary)

# VISUALISASI CLUSTER (2 DIMENSI)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='jumlah',
    y='total',
    hue='cluster',
    data=df,
    palette='Set2'
)
plt.title('Visualisasi Cluster Berdasarkan Jumlah & Total Transaksi')
plt.xlabel('Jumlah Tiket')
plt.ylabel('Total Transaksi')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# SIMPAN DATA DENGAN LABEL CLUSTER

OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "penjualan_tiket_clustered.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"\ Clustering selesai. Data disimpan di:\n{OUTPUT_PATH}")

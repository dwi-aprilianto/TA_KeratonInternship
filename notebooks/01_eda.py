# EXPLORATORY DATA ANALYSIS (EDA)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8')
sns.set_palette('Set2')

# PATH SETUP

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / 'data' / 'raw' / 'penjualan_tiket_raw.xlsx'

print("BASE_DIR    :", BASE_DIR)
print("PROJECT_DIR :", PROJECT_DIR)
print("DATA_PATH   :", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"File Excel tidak ditemukan: {DATA_PATH}")

# LOAD DATASET

df = pd.read_excel(DATA_PATH)

# INFORMASI DATASET

print("\n=== INFORMASI DATASET ===")
df.info()

print("\n=== 5 DATA TERATAS ===")
print(df.head())

print("\nJumlah baris :", df.shape[0])
print("Jumlah kolom :", df.shape[1])

# CEK MISSING VALUES

print("\n=== MISSING VALUES ===")
print(df.isna().sum())

# STATISTIK DESKRIPTIF

print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe())

# DISTRIBUSI JUMLAH TIKET

plt.figure(figsize=(10,5))
sns.histplot(df['Jumlah'], bins=30, kde=True)
plt.title('Distribusi Jumlah Tiket')
plt.xlabel('Jumlah Tiket')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.show()

# BOXPLOT JUMLAH TIKET

plt.figure(figsize=(10,4))
sns.boxplot(x=df['Jumlah'])
plt.title('Boxplot Jumlah Tiket')
plt.xlabel('Jumlah Tiket')
plt.tight_layout()
plt.show()

# DISTRIBUSI TOTAL TRANSAKSI

plt.figure(figsize=(10,5))
sns.histplot(df['Total'], bins=30, kde=True)
plt.title('Distribusi Total Transaksi')
plt.xlabel('Total Transaksi')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.show()

# KONVERSI TANGGAL

df['Tanggal'] = pd.to_datetime(df['Tanggal'], dayfirst=True)

# PENJUALAN PER HARI (EDA TIME)

daily_sales = df.groupby(df['Tanggal'].dt.date)['Jumlah'].sum()

plt.figure(figsize=(12,6))
daily_sales.plot()
plt.title('Penjualan Tiket Harian')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Tiket')
plt.tight_layout()
plt.show()

# HEATMAP KORELASI

plt.figure(figsize=(6,4))
sns.heatmap(
    df[['Jumlah', 'Total']].corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)
plt.title('Korelasi Jumlah Tiket dan Total Transaksi')
plt.tight_layout()
plt.show()

# KESIMPULAN EDA 

print("""
KESIMPULAN SEMENTARA EDA:
1. Jumlah tiket yang terjual memiliki variasi yang cukup besar.
2. Terdapat indikasi outlier pada jumlah tiket dan total transaksi.
3. Penjualan tiket menunjukkan pola fluktuatif dari waktu ke waktu.
4. Variabel jumlah tiket dan total transaksi memiliki korelasi positif.
5. Data layak untuk dilanjutkan ke tahap preprocessing dan clustering.
""")

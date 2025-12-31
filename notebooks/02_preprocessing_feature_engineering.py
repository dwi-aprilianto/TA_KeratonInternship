# 02_preprocessing_feature_engineering.py

import pandas as pd
from pathlib import Path

# PATH SETUP 

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
RAW_PATH = PROJECT_DIR / "data" / "raw" / "penjualan_tiket_raw.xlsx"
PROCESSED_PATH = PROJECT_DIR / "data" / "processed" / "penjualan_tiket_clean.csv"

print("PROJECT_DIR :", PROJECT_DIR)
print("RAW_PATH    :", RAW_PATH)

# LOAD DATA

df = pd.read_excel(RAW_PATH)

print("\n=== DATA AWAL ===")
print(df.head())

# PREPROCESSING

# 1. Rename kolom agar konsisten & aman
df.columns = df.columns.str.lower()

# 2. Konversi kolom tanggal
df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d-%m-%Y %H:%M:%S')

# 3. Pastikan numerik
df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce')
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# 4. Cek missing value
print("\n=== MISSING VALUES SETELAH KONVERSI ===")
print(df.isnull().sum())

# 5. Drop data anomali (jika ada)
df = df.dropna()

# FEATURE ENGINEERING

# 1. Fitur waktu
df['tahun'] = df['tanggal'].dt.year
df['bulan'] = df['tanggal'].dt.month
df['nama_bulan'] = df['tanggal'].dt.month_name()
df['hari'] = df['tanggal'].dt.day
df['hari_dalam_minggu'] = df['tanggal'].dt.day_name()
df['jam'] = df['tanggal'].dt.hour

# 2. Weekend vs Weekday
df['is_weekend'] = df['hari_dalam_minggu'].isin(['Saturday', 'Sunday']).astype(int)

# 3. Harga per tiket
df['harga_per_tiket'] = df['total'] / df['jumlah']

# 4. Label volume penjualan (optional – untuk ML)
df['volume_penjualan'] = pd.cut(
    df['jumlah'],
    bins=[0, 2, 5, 10, 50, 500],
    labels=['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
)

# VALIDASI DATA AKHIR

print("\n=== DATA SETELAH PREPROCESSING ===")
print(df.head())

print("\n=== INFO DATA ===")
print(df.info())

# SAVE CLEAN DATA

df.to_csv(PROCESSED_PATH, index=False)
print(f"\n✅ Data bersih disimpan di: {PROCESSED_PATH}")

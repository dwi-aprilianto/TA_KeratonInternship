# 04_anova_analysis.py
# One-Way ANOVA Penjualan Tiket

import pandas as pd
from scipy.stats import shapiro, f_oneway
from pathlib import Path
import numpy as np

# PATH SETUP

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "penjualan_tiket_clean.csv"

print("DATA_PATH :", DATA_PATH)

# LOAD DATA

df = pd.read_csv(DATA_PATH)

# VALIDASI KOLOM

required_columns = ['jumlah', 'is_weekend']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset")

# PEMISAHAN DATA

weekday = df[df['is_weekend'] == 0]['jumlah']
weekend = df[df['is_weekend'] == 1]['jumlah']

print("\nJumlah data Weekday :", len(weekday))
print("Jumlah data Weekend :", len(weekend))

# UJI NORMALITAS (Shapiro-Wilk)

print("\n=== UJI NORMALITAS (Shapiro-Wilk) ===")

# Sampling jika data terlalu besar (Shapiro max ~5000)
def shapiro_test(data, label):
    sample = data.sample(n=5000, random_state=42) if len(data) > 5000 else data
    stat, p = shapiro(sample)
    print(f"{label}: Statistik={stat:.4f}, p-value={p:.4f}")
    return p

p_weekday = shapiro_test(weekday, "Weekday")
p_weekend = shapiro_test(weekend, "Weekend")

# UJI HOMOGENITAS VARIANS

print("\n=== VARIANS DATA ===")
print("Varians Weekday :", np.var(weekday))
print("Varians Weekend :", np.var(weekend))

# ONE-WAY ANOVA

print("\n=== ONE-WAY ANOVA ===")

anova_stat, anova_p = f_oneway(weekday, weekend)

print(f"F-Statistic : {anova_stat:.4f}")
print(f"p-value     : {anova_p:.6f}")

# INTERPRETASI OTOMATIS

alpha = 0.05

print("\n=== INTERPRETASI ===")

if anova_p < alpha:
    print(
        "Terdapat perbedaan yang SIGNIFIKAN antara "
        "penjualan tiket pada hari kerja (Weekday) "
        "dan akhir pekan (Weekend)."
    )
else:
    print(
        "Tidak terdapat perbedaan yang signifikan antara "
        "penjualan tiket pada hari kerja dan akhir pekan."
    )

print("""ANOVA Analysis selesai dan valid """)

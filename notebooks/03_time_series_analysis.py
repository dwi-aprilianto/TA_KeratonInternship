# 03_time_series_analysis.py
# Analisis Tren Penjualan Tiket

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8')

# PATH SETUP

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "penjualan_tiket_clean.csv"

print("BASE_DIR    :", BASE_DIR)
print("PROJECT_DIR :", PROJECT_DIR)
print("DATA_PATH   :", DATA_PATH)

# LOAD DATA

df = pd.read_csv(DATA_PATH)

# Pastikan kolom tanggal dalam format datetime
df['tanggal'] = pd.to_datetime(df['tanggal'])

print("\n=== INFO DATA ===")
print(df.info())

# AGREGASI DATA TIME SERIES

# Penjualan harian
daily_sales = df.groupby('tanggal')['jumlah'].sum()

# Penjualan mingguan
weekly_sales = (
    df.set_index('tanggal')
      .resample('W')['jumlah']
      .sum()
)

# Penjualan bulanan
monthly_sales = (
    df.set_index('tanggal')
      .resample('M')['jumlah']
      .sum()
)

print("\n=== CONTOH AGREGASI HARIAN ===")
print(daily_sales.head())

# MOVING AVERAGE

daily_ma7 = daily_sales.rolling(window=7).mean()
daily_ma14 = daily_sales.rolling(window=14).mean()

# VISUALISASI TREN HARIAN

plt.figure(figsize=(12, 6))
plt.plot(daily_sales, label='Penjualan Harian', alpha=0.4)
plt.plot(daily_ma7, label='Moving Average 7 Hari', linewidth=2)
plt.plot(daily_ma14, label='Moving Average 14 Hari', linewidth=2)
plt.title('Tren Penjualan Tiket Harian')
plt.xlabel('Tanggal')
plt.ylabel('Jumlah Tiket')
plt.legend()
plt.tight_layout()
plt.show()

# VISUALISASI TREN MINGGUAN

plt.figure(figsize=(12, 6))
plt.plot(weekly_sales, marker='o')
plt.title('Tren Penjualan Tiket Mingguan')
plt.xlabel('Minggu')
plt.ylabel('Jumlah Tiket')
plt.tight_layout()
plt.show()

# VISUALISASI TREN BULANAN

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales, marker='o', linewidth=2)
plt.title('Tren Penjualan Tiket Bulanan')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Tiket')
plt.tight_layout()
plt.show()

print("\n✅ Time Series Analysis selesai")

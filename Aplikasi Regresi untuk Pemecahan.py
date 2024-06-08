import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Path ke file CSV
file_path = r'C:\Users\HP\Desktop\csvv\Student_Performance.csv'

# Memuat data dari file CSV
df = pd.read_csv(file_path)

# Ekstraksi kolom yang diperlukan
TB = df['Hours Studied'].values
NL = df['Sample Question Papers Practiced'].values
NT = df['Performance Index'].values

# Problem 2 dengan Metode 1 (Model Linear)
X = NL.reshape(-1, 1)
linear_model = LinearRegression()
linear_model.fit(X, NT)

# Prediksi menggunakan model linear
NT_pred_linear = linear_model.predict(X)

# Plot hasil regresi linear
plt.scatter(NL, NT, color='blue', label='Data Asli')
plt.plot(NL, NT_pred_linear, color='red', label='Regresi Linear')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Linear')
plt.legend()
plt.show()

# Hitung galat RMS untuk model linear
rms_linear = np.sqrt(mean_squared_error(NT, NT_pred_linear))
print(f'RMS (Model Linear): {rms_linear}')

# Problem 2 dengan Metode 2 (Model Pangkat Sederhana)
def power_law(x, a, b):
    return a * np.power(x, b)

# Fit model pangkat sederhana
params, covariance = curve_fit(power_law, NL, NT)

# Prediksi menggunakan model pangkat sederhana
NT_pred_power = power_law(NL, *params)

# Plot hasil regresi pangkat sederhana
plt.scatter(NL, NT, color='blue', label='Data Asli')
plt.plot(NL, NT_pred_power, color='red', label='Regresi Pangkat Sederhana')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.title('Regresi Pangkat Sederhana')
plt.legend()
plt.show()

# Hitung galat RMS untuk model pangkat sederhana
rms_power = np.sqrt(mean_squared_error(NT, NT_pred_power))
print(f'RMS (Model Pangkat Sederhana): {rms_power}')

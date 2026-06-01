import pandas as pd
import numpy as np

# 1. Load Data & Transformasi Matriks User-Item
df_filtered = pd.read_csv('../dataset/rating_final.csv')[['userID', 'placeID', 'food_rating']]
matrix_df = df_filtered.pivot_table(index='userID', columns='placeID', values='food_rating').fillna(0)
R = matrix_df.to_numpy()

print(f"Bentuk Matriks: {R.shape} | Total Rating: {np.count_nonzero(R)}\n")

# 2. Setup Variabel Target (Contoh: User index 0)
target_user_idx = 0
b = R[target_user_idx, :].T

# Matriks A: Fitur dari rating seluruh user lain (Transpose ke Item x User Lain)
A = np.delete(R, target_user_idx, axis=0).T 

# 3. Hitung Least Square: (A^T * A)x = A^T * b
At_A = np.dot(A.T, A)
At_b = np.dot(A.T, b)

# Tambahkan regularisasi kecil agar matriks tidak singular saat dihitung
At_A_reg = At_A + np.eye(At_A.shape[0]) * 0.01 
x = np.linalg.solve(At_A_reg, At_b)

# Prediksi rating penuh untuk target user
predicted_b = np.dot(A, x)

# 4. Filter & Ekstraksi Top-5 Rekomendasi Menu Baru
prediksi_semua = predicted_b.flatten()
menu_belum_dicoba_idx = np.where(b == 0)[0]
prediksi_menu_baru = prediksi_semua[menu_belum_dicoba_idx]

# Urutkan indeks dari prediksi rating tertinggi
rekomendasi_idx = np.argsort(prediksi_menu_baru)[::-1]

data_ekspor = []
print("--- 🌟 TOP 5 REKOMENDASI MENU UNTUK USER 1 🌟 ---")

for i in range(5):
    idx_asli = menu_belum_dicoba_idx[rekomendasi_idx[i]]
    skor = prediksi_menu_baru[rekomendasi_idx[i]]
    place_id = matrix_df.columns[idx_asli] 
    
    # Hati-hati, tanda kurung di baris print dan append ini jangan sampai kepotong
    print(f"{i+1}. Place ID: {place_id} (Prediksi Rating: {skor:.4f})")
    
    data_ekspor.append({
        'Ranking': i + 1,
        'Place_ID': place_id,
        'Prediksi_Rating': round(skor, 4)
    })

# 5. Export hasil ke format CSV
pd.DataFrame(data_ekspor).to_csv('hasil_rekomendasi.csv', index=False)
print("\n[SUCCESS] File 'hasil_rekomendasi.csv' berhasil dibuat!")
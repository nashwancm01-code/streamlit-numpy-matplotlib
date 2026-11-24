#double pendulum nashwa nurcinta 5023241060
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
# Mengatur judul tab browser dan tata letak lebar agar grafik & animasi muat rapi.
st.set_page_config(page_title="Simulasi Double Pendulum", layout="wide")
st.title("Simulasi Double Pendulum (Lagrange & RK4 From Scratch)")
st.caption("Metode Lagrange untuk persamaan gerak & Runge-Kutta Orde 4 untuk integrasi numerik")


# ==============================================================================
# 2. FUNGSI TURUNAN SISTEM (PERSAMAAN GERAK DARI LAGRANGE)
# ==============================================================================
# State berisi 4 variabel: [theta1, omega1, theta2, omega2]
# - theta1, theta2 : posisi sudut batang 1 & 2 (dalam radian)
# - omega1, omega2 : kecepatan sudut (theta1_dot dan theta2_dot, dalam rad/s)
#
# Dari persamaan Lagrange yang sudah diturunkan:
#   M11 * alpha1 + M12 * alpha2 = rhs1
#   M21 * alpha1 + M22 * alpha2 = rhs2
# Di mana alpha1 = theta1_ddot dan alpha2 = theta2_ddot (percepatan sudut).
def hitung_turunan(state, m1, m2, l1, l2, I1, I2, g):
    theta1, omega1, theta2, omega2 = state

    # Selisih sudut antar batang sangat sering muncul di suku kopel (gaya/inersia silang)
    delta = theta1 - theta2
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    # Koefisien massa/inersia efektif (bagian kiri persamaan matriks)
    M11 = (0.25 * m1 * l1**2) + (m2 * l1**2) + I1
    M12 = 0.5 * m2 * l1 * l2 * cos_d
    M21 = 0.5 * m2 * l1 * l2 * cos_d
    M22 = (0.25 * m2 * l2**2) + I2

    # Sisi kanan (memindahkan suku sentripetal/coriolis dan gravitasi ke ruas kanan)
    # Tanda minus pada gravitasi muncul karena definisi h = l1 + l2 - (...) pada geometri slide
    rhs1 = -(0.5 * m2 * l1 * l2 * (omega2**2) * sin_d) - ((0.5 * m1 + m2) * g * l1 * np.sin(theta1))
    rhs2 = (0.5 * m2 * l1 * l2 * (omega1**2) * sin_d) - (0.5 * m2 * g * l2 * np.sin(theta2))

    # Menyelesaikan sistem persamaan linear 2x2 secara manual (Aturan Cramer)
    # untuk mendapatkan percepatan sudut alpha1 (theta1_ddot) dan alpha2 (theta2_ddot)
    det = (M11 * M22) - (M12 * M21)
    alpha1 = ((rhs1 * M22) - (M12 * rhs2)) / det
    alpha2 = ((M11 * rhs2) - (rhs1 * M21)) / det

    # Turunan dari [theta1, omega1, theta2, omega2] adalah [omega1, alpha1, omega2, alpha2]
    return np.array([omega1, alpha1, omega2, alpha2], dtype=float)


# ==============================================================================
# 3. METODE INTEGRASI NUMERIK RUNGE-KUTTA ORDE 4 (RK4) MANUAL
# ==============================================================================
# Menghitung nilai state pada langkah waktu berikutnya (t + dt) menggunakan k1, k2, k3, k4
# agar simulasi sistem non-linear seperti double pendulum tetap stabil.
def langkah_rk4(state, dt, m1, m2, l1, l2, I1, I2, g):
    k1 = hitung_turunan(state, m1, m2, l1, l2, I1, I2, g)
    k2 = hitung_turunan(state + 0.5 * dt * k1, m1, m2, l1, l2, I1, I2, g)
    k3 = hitung_turunan(state + 0.5 * dt * k2, m1, m2, l1, l2, I1, I2, g)
    k4 = hitung_turunan(state + dt * k3, m1, m2, l1, l2, I1, I2, g)

    state_berikutnya = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state_berikutnya


# ==============================================================================
# 4. PANEL INPUT PARAMETER DI SIDEBAR
# ==============================================================================
st.sidebar.header("Parameter Simulasi")

# Massa dan panjang batang (mengacu ke contoh input di slide simulasi)
m1 = st.sidebar.number_input("Massa Batang 1, m1 (kg)", min_value=0.1, value=10.0, step=0.5)
m2 = st.sidebar.number_input("Massa Batang 2, m2 (kg)", min_value=0.1, value=10.0, step=0.5)
l1 = st.sidebar.number_input("Panjang Batang 1, l1 (m)", min_value=0.1, value=5.0, step=0.5)
l2 = st.sidebar.number_input("Panjang Batang 2, l2 (m)", min_value=0.1, value=5.0, step=0.5)

# Sudut awal dalam derajat (akan dikonversi ke radian untuk perhitungan trigonometri)
theta1_deg = st.sidebar.number_input("Sudut Awal Theta 1 (deg)", value=30.0, step=5.0)
theta2_deg = st.sidebar.number_input("Sudut Awal Theta 2 (deg)", value=10.0, step=5.0)

# Kecepatan sudut awal (default 0 rad/s, dilepas dari diam)
omega1_0 = st.sidebar.number_input("Kecepatan Sudut Awal 1 (rad/s)", value=0.0, step=0.5)
omega2_0 = st.sidebar.number_input("Kecepatan Sudut Awal 2 (rad/s)", value=0.0, step=0.5)

# Frekuensi sampling & durasi simulasi (misal 100 Hz -> dt = 0.01 s)
fs = st.sidebar.number_input("Sampling Rate (Hz)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
durasi = st.sidebar.slider("Durasi Simulasi (detik)", min_value=2.0, max_value=25.0, value=10.0, step=1.0)
g = st.sidebar.number_input("Gravitasi, g (m/s^2)", min_value=0.1, value=9.81, step=0.01)

# Momen inersia batang homogen di pusat massa: I = (1/12) * m * l^2
I1 = (1.0 / 12.0) * m1 * (l1**2)
I2 = (1.0 / 12.0) * m2 * (l2**2)


# ==============================================================================
# 5. LOOP SIMULASI DARI AWAL SAMPAI AKHIR WAKTU
# ==============================================================================
dt = 1.0 / fs
jumlah_langkah = int(durasi / dt)

# Konversi sudut awal derajat ke radian
state_sekarang = np.array([
    np.deg2rad(theta1_deg),
    omega1_0,
    np.deg2rad(theta2_deg),
    omega2_0
], dtype=float)

# Menyimpan riwayat waktu dan sudut untuk plotting grafik
waktu = np.zeros(jumlah_langkah + 1)
riwayat_theta1_rad = np.zeros(jumlah_langkah + 1)
riwayat_theta2_rad = np.zeros(jumlah_langkah + 1)
riwayat_theta1_deg = np.zeros(jumlah_langkah + 1)
riwayat_theta2_deg = np.zeros(jumlah_langkah + 1)

waktu[0] = 0.0
riwayat_theta1_rad[0] = state_sekarang[0]
riwayat_theta2_rad[0] = state_sekarang[2]
riwayat_theta1_deg[0] = np.rad2deg(state_sekarang[0])
riwayat_theta2_deg[0] = np.rad2deg(state_sekarang[2])

# Melakukan iterasi RK4 langkah demi langkah
for i in range(1, jumlah_langkah + 1):
    state_sekarang = langkah_rk4(state_sekarang, dt, m1, m2, l1, l2, I1, I2, g)
    waktu[i] = i * dt
    riwayat_theta1_rad[i] = state_sekarang[0]
    riwayat_theta2_rad[i] = state_sekarang[2]
    riwayat_theta1_deg[i] = np.rad2deg(state_sekarang[0])
    riwayat_theta2_deg[i] = np.rad2deg(state_sekarang[2])


# ==============================================================================
# 6. VISUALISASI GRAFIK PERUBAHAN SUDUT TERHADAP WAKTU
# ==============================================================================
satuan_grafik = st.radio("Satuan sudut pada grafik:", ["Derajat (deg)", "Radian (rad)"], horizontal=True)

if satuan_grafik == "Derajat (deg)":
    y1, y2, label_y = riwayat_theta1_deg, riwayat_theta2_deg, "Sudut (deg)"
else:
    y1, y2, label_y = riwayat_theta1_rad, riwayat_theta2_rad, "Sudut (rad)"

kol_kiri, kol_kanan = st.columns([2, 1])

with kol_kiri:
    st.subheader("Grafik Perubahan Sudut Terhadap Waktu")
    fig_grafik, ax_grafik = plt.subplots(figsize=(8, 4))
    ax_grafik.plot(waktu, y1, label="Theta 1 (Batang 1)", color="blue")
    ax_grafik.plot(waktu, y2, label="Theta 2 (Batang 2)", color="green")
    ax_grafik.set_xlabel("Waktu (s)")
    ax_grafik.set_ylabel(label_y)
    ax_grafik.grid(True, linestyle="--", alpha=0.5)
    ax_grafik.legend()
    st.pyplot(fig_grafik)

with kol_kanan:
    st.subheader("Cuplikan Nilai Sudut (rad)")
    # Menampilkan beberapa baris awal hasil perhitungan numerik
    tampilan_data = [
        {"t (s)": round(waktu[idx], 2), "theta1 (rad)": round(riwayat_theta1_rad[idx], 4), "theta2 (rad)": round(riwayat_theta2_rad[idx], 4)}
        for idx in range(0, len(waktu), max(1, int(fs // 10)))
    ]
    st.dataframe(tampilan_data, height=280, use_container_width=True)


# ==============================================================================
# 7. VISUALISASI POSISI / ANIMASI DOUBLE PENDULUM
# ==============================================================================
# Mengubah sudut theta1 dan theta2 menjadi koordinat ujung batang (x1, y1) dan (x2, y2)
# Titik tumpuan (pivot) diletakkan di (0, 0), arah ke bawah bernilai negatif pada sumbu y.
x1_all = l1 * np.sin(riwayat_theta1_rad)
y1_all = -l1 * np.cos(riwayat_theta1_rad)
x2_all = x1_all + l2 * np.sin(riwayat_theta2_rad)
y2_all = y1_all - l2 * np.cos(riwayat_theta2_rad)

st.subheader("Visualisasi Posisi Double Pendulum")
mode_tampil = st.radio("Mode tampilan bandul:", ["Snapshot pada waktu tertentu", "Putar animasi otomatis"], horizontal=True)

batas_sumbu = (l1 + l2) * 1.15

if mode_tampil == "Snapshot pada waktu tertentu":
    # Memilih indeks frame lewat slider agar bisa mengamati posisi bandul di detik mana saja
    idx_pilih = st.slider("Geser untuk memilih langkah waktu:", min_value=0, max_value=jumlah_langkah, value=0)
    st.write(f"Waktu: **{waktu[idx_pilih]:.2f} s** | Theta 1: **{riwayat_theta1_deg[idx_pilih]:.2f}°** | Theta 2: **{riwayat_theta2_deg[idx_pilih]:.2f}°**")

    fig_bandul, ax_bandul = plt.subplots(figsize=(5, 5))
    # Jejak lintasan ujung batang kedua (opsional, membantu melihat gerak chaos/ayunan)
    ax_bandul.plot(x2_all[:idx_pilih + 1], y2_all[:idx_pilih + 1], color="lightgray", linewidth=1, label="Jejak ujung 2")
    # Batang 1 dan Batang 2
    ax_bandul.plot([0, x1_all[idx_pilih], x2_all[idx_pilih]], [0, y1_all[idx_pilih], y2_all[idx_pilih]], marker="o", markersize=8, linewidth=2.5, color="crimson")
    ax_bandul.plot(0, 0, marker="v", markersize=10, color="black")  # Titik tumpuan
    ax_bandul.set_xlim(-batas_sumbu, batas_sumbu)
    ax_bandul.set_ylim(-batas_sumbu, batas_sumbu)
    ax_bandul.set_aspect("equal")
    ax_bandul.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig_bandul)

else:
    # Animasi sederhana dengan memperbarui placeholder plot secara bertahap
    # Lompatan frame diatur agar tidak terlalu lambat saat di-render di browser
    langkah_frame = max(1, int(fs // 25))
    tempat_plot = st.empty()
    info_waktu = st.empty()

    for idx in range(0, jumlah_langkah + 1, langkah_frame):
        fig_anim, ax_anim = plt.subplots(figsize=(5, 5))
        ax_anim.plot(x2_all[:idx + 1], y2_all[:idx + 1], color="lightgray", linewidth=1)
        ax_anim.plot([0, x1_all[idx], x2_all[idx]], [0, y1_all[idx], y2_all[idx]], marker="o", markersize=8, linewidth=2.5, color="crimson")
        ax_anim.plot(0, 0, marker="v", markersize=10, color="black")
        ax_anim.set_xlim(-batas_sumbu, batas_sumbu)
        ax_anim.set_ylim(-batas_sumbu, batas_sumbu)
        ax_anim.set_aspect("equal")
        ax_anim.grid(True, linestyle="--", alpha=0.4)
        tempat_plot.pyplot(fig_anim)
        plt.close(fig_anim)
        info_waktu.caption(f"Waktu simulasi: {waktu[idx]:.2f} s")

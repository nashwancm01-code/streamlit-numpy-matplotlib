#double pendulum nashwa nurcinta 5023241060
import time
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ==============================================================================
# 1. KONFIGURASI HALAMAN & JUDUL (MENGACU TAMPILAN SLIDE 19)
# ==============================================================================
st.set_page_config(page_title="Simulasi Double Pendulum", layout="wide")
st.title("Simulasi Double Pendulum")
st.caption("Persamaan gerak berbasis Metode Lagrange & integrasi numerik Runge-Kutta Orde 4 (tanpa gaya gesek)")


# ==============================================================================
# 2. FUNGSI PERSAMAAN GERAK DARI METODE LAGRANGE (FROM SCRATCH)
# ==============================================================================
# Vektor state berisi 4 nilai: [theta1, omega1, theta2, omega2]
# - theta1, theta2 : sudut simpangan batang 1 & 2 (radian)
# - omega1, omega2 : kecepatan sudut (d(theta)/dt dalam rad/s)
#
# Dari penurunan Lagrange (L = Ek - Ep) untuk double pendulum dengan pusat massa
# di tengah batang (l1/2 dan l2/2) serta momen inersia I1 dan I2:
#   M11 * alpha1 + M12 * alpha2 = rhs1
#   M21 * alpha1 + M22 * alpha2 = rhs2
def hitung_turunan_lagrange(state, m1, m2, l1, l2, I1, I2, g):
    theta1, omega1, theta2, omega2 = state

    # Selisih sudut antar batang muncul pada suku kopel (interaksi gerak batang 1 & 2)
    delta = theta1 - theta2
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)

    # Elemen matriks massa/inersia efektif (ruas kiri persamaan gerak)
    M11 = (0.25 * m1 * (l1 ** 2)) + (m2 * (l1 ** 2)) + I1
    M12 = 0.5 * m2 * l1 * l2 * cos_d
    M21 = 0.5 * m2 * l1 * l2 * cos_d
    M22 = (0.25 * m2 * (l2 ** 2)) + I2

    # Suku gaya/torsi gravitasi dan sentripetal/coriolis (ruas kanan, torsi luar = 0)
    rhs1 = -(0.5 * m2 * l1 * l2 * (omega2 ** 2) * sin_d) - ((0.5 * m1 + m2) * g * l1 * np.sin(theta1))
    rhs2 = (0.5 * m2 * l1 * l2 * (omega1 ** 2) * sin_d) - (0.5 * m2 * g * l2 * np.sin(theta2))

    # Menyelesaikan sistem persamaan linear 2x2 manual (Aturan Cramer)
    # untuk memperoleh percepatan sudut alpha1 (theta1_ddot) dan alpha2 (theta2_ddot)
    det = (M11 * M22) - (M12 * M21)
    alpha1 = ((rhs1 * M22) - (M12 * rhs2)) / det
    alpha2 = ((M11 * rhs2) - (rhs1 * M21)) / det

    # Mengembalikan turunan pertama dari [theta1, omega1, theta2, omega2]
    return np.array([omega1, alpha1, omega2, alpha2], dtype=float)


# ==============================================================================
# 3. METODE RUNGE-KUTTA ORDE 4 (RK4) MANUAL PER LANGKAH WAKTU dt
# ==============================================================================
def rk4_satu_langkah(state, dt, m1, m2, l1, l2, I1, I2, g):
    # Menghitung 4 slope (k1, k2, k3, k4) sesuai rumus RK4 dari materi
    k1 = hitung_turunan_lagrange(state, m1, m2, l1, l2, I1, I2, g)
    k2 = hitung_turunan_lagrange(state + 0.5 * dt * k1, m1, m2, l1, l2, I1, I2, g)
    k3 = hitung_turunan_lagrange(state + 0.5 * dt * k2, m1, m2, l1, l2, I1, I2, g)
    k4 = hitung_turunan_lagrange(state + dt * k3, m1, m2, l1, l2, I1, I2, g)

    # Menggabungkan k1..k4 untuk mendapatkan state pada waktu berikutnya
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ==============================================================================
# 4. PANEL PARAMETER & KONTROL (SEPERTI INPUT DI SLIDE 19 + WAKTU BERHENTI)
# ==============================================================================
st.sidebar.header("Parameter & Kontrol")

# Nilai default mengikuti contoh di Slide 19 (m1=10, m2=10, l1=5, l2=5, teta1=30, teta2=10)
m1 = st.sidebar.number_input("m1 (kg)", min_value=0.1, value=10.0, step=1.0)
m2 = st.sidebar.number_input("m2 (kg)", min_value=0.1, value=10.0, step=1.0)
l1 = st.sidebar.number_input("l1 (m)", min_value=0.1, value=5.0, step=1.0)
l2 = st.sidebar.number_input("l2 (m)", min_value=0.1, value=5.0, step=1.0)

teta1_deg = st.sidebar.number_input("teta1 awal (deg)", value=30.0, step=5.0)
teta2_deg = st.sidebar.number_input("teta2 awal (deg)", value=10.0, step=5.0)

# Slider frekuensi (Hz) untuk mengatur kerapatan sampling & kecepatan laju simulasi
frekuensi_hz = st.sidebar.slider("Frekuensi / Kecepatan (Hz)", min_value=20, max_value=200, value=100, step=10)

# Waktu target berhenti otomatis (misal 35 detik sesuai contoh temanmu)
waktu_berhenti = st.sidebar.number_input("Berhenti di detik ke- (s)", min_value=1.0, max_value=120.0, value=35.0, step=1.0)

# Gravitasi standar (bisa disesuaikan jika perlu, default 9.81 m/s^2)
g = st.sidebar.number_input("Gravitasi g (m/s^2)", min_value=0.1, value=9.81, step=0.01)

# Momen inersia batang homogen di pusat massa: I = (1/12) * m * l^2
I1 = (1.0 / 12.0) * m1 * (l1 ** 2)
I2 = (1.0 / 12.0) * m2 * (l2 ** 2)
dt = 1.0 / float(frekuensi_hz)

# Kunci konfigurasi untuk mendeteksi kalau user mengganti parameter input
config_kunci = (m1, m2, l1, l2, teta1_deg, teta2_deg, frekuensi_hz, waktu_berhenti, g)


def reset_simulasi():
    # Menginisialisasi ulang kondisi awal dari t = 0 detik
    th1_0 = float(np.deg2rad(teta1_deg))
    th2_0 = float(np.deg2rad(teta2_deg))
    st.session_state.config_kunci = config_kunci
    st.session_state.state_curr = np.array([th1_0, 0.0, th2_0, 0.0], dtype=float)
    st.session_state.t_curr = 0.0
    st.session_state.running = True  # Langsung siap gerak otomatis saat Start / awal buka
    st.session_state.hist_t = [0.0]
    st.session_state.hist_th1 = [th1_0]
    st.session_state.hist_th2 = [th2_0]


# Inisialisasi pertama kali atau reset otomatis jika parameter diubah
if "state_curr" not in st.session_state or st.session_state.config_kunci != config_kunci:
    reset_simulasi()


# ==============================================================================
# 5. TOMBOL KONTROL (START / CONTINUE / PAUSE / RESET)
# ==============================================================================
kol_btn1, kol_btn2, kol_btn3, kol_info = st.columns([1, 1, 1, 3])

with kol_btn1:
    if st.button("Start", use_container_width=True):
        # Jika sudah sampai waktu_berhenti, Start akan mengulang dari 0; jika di tengah, lanjutkan
        if st.session_state.t_curr >= waktu_berhenti:
            reset_simulasi()
        st.session_state.running = True

with kol_btn2:
    if st.button("Continue / Pause", use_container_width=True):
        # Toggle antara jalan (True) dan jeda (False), selama belum melewati batas waktu
        if st.session_state.t_curr < waktu_berhenti:
            st.session_state.running = not st.session_state.running

with kol_btn3:
    if st.button("Reset", use_container_width=True):
        reset_simulasi()

with kol_info:
    status_teks = "Berjalan" if st.session_state.running else ("Selesai / Berhenti" if st.session_state.t_curr >= waktu_berhenti else "Dijeda")
    st.markdown(f"**Status:** {status_teks} | **Waktu:** `{st.session_state.t_curr:.2f} s` / `{waktu_berhenti:.1f} s`")


# ==============================================================================
# 6. AREA TAMPILAN UTAMA (GRAFIK KIRI & ANIMASI BANDUL KANAN SEPERTI SLIDE 19)
# ==============================================================================
area_grafik_kiri, area_bandul_kanan = st.columns([1.25, 1.0])

with area_grafik_kiri:
    st.subheader("Grafik Perubahan Posisi Terhadap Waktu")
    slot_grafik = st.empty()

with area_bandul_kanan:
    st.subheader("Simulasi Double Pendulum")
    slot_bandul = st.empty()


# ==============================================================================
# 7. DAFTAR NILAI TETA1 & TETA2 (DALAM RADIAN, SEPERTI LISTBOX DI SLIDE 19)
# ==============================================================================
st.markdown("### Nilai Sudut (radian)")
kol_teta1, kol_teta2 = st.columns(2)
with kol_teta1:
    st.caption("Teta1 (rad)")
    slot_list1 = st.empty()
with kol_teta2:
    st.caption("Teta2 (rad)")
    slot_list2 = st.empty()


def gambar_frame(t_arr, th1_arr, th2_arr):
    # --- Gambar grafik posisi sudut (rad) terhadap waktu ---
    fig_g, ax_g = plt.subplots(figsize=(6.5, 3.8))
    ax_g.plot(t_arr, th1_arr, color="blue", linewidth=1.3, label="Teta1 (rad)")
    ax_g.plot(t_arr, th2_arr, color="limegreen", linewidth=1.3, label="Teta2 (rad)")
    ax_g.set_xlim(0, max(waktu_berhenti, t_arr[-1] if len(t_arr) > 0 else 1.0))
    ax_g.set_xlabel("Time [s]")
    ax_g.set_ylabel("Sudut [rad]")
    ax_g.grid(True, linestyle="--", alpha=0.5)
    ax_g.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    slot_grafik.pyplot(fig_g)
    plt.close(fig_g)

    # --- Gambar posisi bandul 2D (tumpuan segitiga, batang 1 hijau, batang 2 biru) ---
    th1_now = th1_arr[-1]
    th2_now = th2_arr[-1]

    # Koordinat tumpuan di (0, 0); arah ke bawah negatif pada sumbu y
    x1 = l1 * np.sin(th1_now)
    y1 = -l1 * np.cos(th1_now)
    x2 = x1 + l2 * np.sin(th2_now)
    y2 = y1 - l2 * np.cos(th2_now)

    batas = (l1 + l2) * 1.15
    fig_b, ax_b = plt.subplots(figsize=(5.2, 4.8))
    # Batang 1 (hijau)
    ax_b.plot([0, x1], [0, y1], color="limegreen", linewidth=3)
    ax_b.plot(x1, y1, marker="o", markersize=8, color="limegreen")
    # Batang 2 (biru)
    ax_b.plot([x1, x2], [y1, y2], color="blue", linewidth=3)
    ax_b.plot(x2, y2, marker="o", markersize=8, color="blue")
    # Tumpuan (segitiga hitam di atas)
    ax_b.plot(0, 0, marker="v", markersize=10, color="black")

    ax_b.set_xlim(-batas, batas)
    ax_b.set_ylim(-batas, batas)
    ax_b.set_aspect("equal")
    ax_b.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    slot_bandul.pyplot(fig_b)
    plt.close(fig_b)

    # --- Update daftar nilai Teta1 & Teta2 (menampilkan data terbaru di atas) ---
    cuplikan_1 = [f"{val:.15f}" for val in reversed(th1_arr[-80:])]
    cuplikan_2 = [f"{val:.15f}" for val in reversed(th2_arr[-80:])]
    slot_list1.code("\n".join(cuplikan_1), language=None)
    slot_list2.code("\n".join(cuplikan_2), language=None)


# ==============================================================================
# 8. LOOP ANIMASI OTOMATIS SAMPAI WAKTU BERHENTI YANG DITENTUKAN
# ==============================================================================
# Menampilkan frame awal dulu
gambar_frame(
    np.array(st.session_state.hist_t),
    np.array(st.session_state.hist_th1),
    np.array(st.session_state.hist_th2),
)

# Jika sedang berjalan dan belum mencapai waktu_berhenti, majukan simulasi secara bertahap
# agar animasi terlihat bergerak di browser dan berhenti otomatis saat t >= waktu_berhenti.
if st.session_state.running and st.session_state.t_curr < waktu_berhenti:
    # Menyesuaikan kecepatan animasi real-time dengan slider frekuensi (Hz)
    jeda_detik = max(0.01, min(0.08, 1.0 / float(frekuensi_hz)))
    # Melangkah beberapa sub-step RK4 per frame agar simulasi 35 detik tidak makan waktu terlalu lama
    sub_steps = max(1, int(round(0.05 / dt)))

    while st.session_state.running and st.session_state.t_curr < waktu_berhenti:
        for _ in range(sub_steps):
            if st.session_state.t_curr >= waktu_berhenti:
                break
            langkah_dt = min(dt, waktu_berhenti - st.session_state.t_curr)
            st.session_state.state_curr = rk4_satu_langkah(
                st.session_state.state_curr, langkah_dt, m1, m2, l1, l2, I1, I2, g
            )
            st.session_state.t_curr += langkah_dt
            st.session_state.hist_t.append(st.session_state.t_curr)
            st.session_state.hist_th1.append(float(st.session_state.state_curr[0]))
            st.session_state.hist_th2.append(float(st.session_state.state_curr[2]))

        gambar_frame(
            np.array(st.session_state.hist_t),
            np.array(st.session_state.hist_th1),
            np.array(st.session_state.hist_th2),
        )

        # Jika sudah menyentuh waktu_berhenti, hentikan otomatis
        if st.session_state.t_curr >= waktu_berhenti:
            st.session_state.running = False
            break

        time.sleep(jeda_detik)

"""Seluruh parameter eksperimen (satu-satunya sumber nilai parameter)."""
# --- Pencuplikan dan episode
DT = 1.0                      # detik (1 Hz)
EPISODE_LEN = 3600            # detik per episode
MAIN_SEED, TUNE_SEED, FOLD_SEED = 42, 7, 42
N_EPISODES, N_LEAK, N_SMALL = 100, 60, 20          # dataset utama
TUNE_EPISODES, TUNE_LEAK, TUNE_SMALL = 40, 24, 8   # dataset penyetelan (seed berbeda)
SMALL_RANGE = (0.10, 0.30)    # bar, penurunan tekanan tunak di sensor 1
LARGE_RANGE = (0.30, 2.00)
ONSET_RANGE = (600, 2400)     # detik, awal kebocoran dalam episode
DURATION_RANGE = (600, 1200)  # detik, lama kebocoran sebelum diperbaiki
TAU_LEAK = 20.0               # detik, konstanta waktu efek kebocoran
GAMMA_RANGE = (0.5, 0.9)      # rasio penurunan sensor 2 terhadap sensor 1
LAG2_RANGE = (3, 10)          # detik, tunda efek kebocoran di sensor 2
# --- Model fisika aliran gas (tunak isotermal, bentuk Weymouth)
PIN0, Q0, K_PIPE = 20.0, 1.0, 144.0   # bar, demand ternormalisasi, bar^2
X_SENSOR = (0.3, 0.7)         # posisi sensor sebagai fraksi panjang pipa
TAU_TRUE, TAU_TWIN = 30.0, 15.0       # detik, lag dinamik sistem vs digital twin
KAPPA_SD = 0.005              # galat relatif K (ketidaksesuaian model) per episode
# --- Derau sensor (bar): AR(1) + putih
NOISE_AR_STD, NOISE_AR_RHO, NOISE_WHITE_STD = 0.03, 0.90, 0.02
# --- Transien operasional
N_TRANSIENTS = (2, 5)         # per episode (inklusif)
PIN_STEP = (0.3, 1.2)         # bar, tanda acak
PIN_RAMP_TOTAL, PIN_RAMP_DUR = (0.5, 1.5), (120, 400)
Q_STEP, Q_PULSE_MULT, Q_PULSE_DUR = (0.75, 1.3), (0.6, 1.4), (20, 80)
# --- Jendela dan evaluasi
WINDOW, STRIDE = 60, 10       # detik
GRACE = 90                    # detik setelah perbaikan, dikeluarkan dari evaluasi
MERGE_GAP = 60                # detik, alarm berjarak <= ini dihitung satu episode alarm
THRESH, NCONSEC = 0.20, 3     # aturan ambang residual (bar, sampel berturut-turut)
N_FOLDS = 5
NOISE_LEVELS = (1, 2, 3)
# --- Simulasi dashboard
DASH_TAU_VALVE, DASH_GAMMA, DASH_LAG2 = 6.0, 0.8, 3
DASH_LEAK_AT, DASH_T, DASH_RUNS = 60, 660, 200
DASH_NORMAL_LEN = 3600
DROP_10, DROP_100 = 0.30, 2.00           # bar di sensor 1 pada bukaan 10% dan 100%
OPENINGS = tuple(range(10, 101, 10))
DASH_THRESHOLDS = (0.10, 0.20, 0.30)
MQTT_LAT_MEDIAN_MS, MQTT_LAT_SIGMA, MQTT_LAT_CLIP = 41.0, 0.30, (18.0, 70.0)

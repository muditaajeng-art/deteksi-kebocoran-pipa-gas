# Deteksi kebocoran pipa gas: machine learning dengan digital twin (simulasi)

Kode eksperimen untuk naskah "Analisis Performa Algoritma Machine Learning dalam Mendeteksi Kebocoran
Pipa Gas Berdasarkan Data Sensor Tekanan dengan Pendekatan Digital Twin: Studi Simulasi".
Seluruh data dibangkitkan oleh simulasi; tidak ada data lapangan.

## Struktur
| Berkas | Isi |
|---|---|
| `config.py` | Semua parameter eksperimen (satu-satunya sumber nilai) |
| `sim.py` | Generator data: model aliran gas tunak + lag dinamik, digital twin, transien, kebocoran, derau |
| `features.py` | Ekstraksi 19 fitur per jendela 60 detik dan aturan ambang residual |
| `run_experiment.py` | Penyetelan hiperparameter, validasi silang 5-lipatan berkelompok per episode, uji derau, benchmark inferensi |
| `run_dashboard.py` | Simulasi deteksi berbasis residual pada digital twin (aturan ambang residual) dan Gambar 2-4 |
| `analysis_baseline_delay.py` | Keterlambatan aturan ambang menurut ukuran kebocoran (mendukung Subbab 4.3) |
| `results/` | `experiment.json`, `dashboard.json`, `baseline_delay.json` dan log hasil yang dilaporkan di naskah |
| `figures/` | Gambar 2-4 naskah |

## Menjalankan ulang
```bash
pip install -r requirements.txt
python run_experiment.py          # sekitar 9 menit pada 2 inti CPU
python run_dashboard.py           # sekitar 2 menit
python analysis_baseline_delay.py
```
Seed: dataset utama 42, dataset penyetelan 7, penetapan lipatan 42, `random_state` model 42.
Metrik klasifikasi dan hasil simulasi deteksi digital twin deterministik; waktu inferensi/latih bergantung pada perangkat.

## Lisensi dan sitasi
Kode dilisensikan di bawah lisensi MIT (lihat berkas `LICENSE`).
DOI Zenodo: ISI_DOI_SETELAH_RILIS

# ЛР 11, вариант 2: рисунки для отчёта по выгруженным из Engee результатам (локально, matplotlib).
# Запуск из каталога LR11:  python src/lr11_figures.py
# Читает artifacts/engee/demo/demo11.json (обучение под наблюдением, seed 101) и artifacts/engee/results_v02.json (10 обучений),
# пишет reports/figures/losses_seed101.png, losses_all_seeds.png, snapshots_A_B.png.
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

E, F = "artifacts/engee", "reports/figures"
demo = json.load(open(f"{E}/demo/demo11.json", encoding="utf-8"))
res = json.load(open(f"{E}/results_v02.json", encoding="utf-8"))
COL = dict(A="#1f5fa8", B="#d0661d")


def smooth(v, w=10):
    """Скользящее среднее по w точкам журнала (журнал пишется каждые 10 шагов, w = 10 — 100 шагов)."""
    v = np.asarray(v, float)
    return np.convolve(v, np.ones(w) / w, mode="valid")


# 1. Обучение под наблюдением (demo_gan.py): потери и средние выходы D, seed 101
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for name in ("A", "B"):
    h = demo[name]["hist"]
    x = np.arange(len(h["lg"])) * 10
    xs = x[9:]
    ax[0].plot(xs, smooth(h["lg"]), color=COL[name], label=f"{name}: потеря G")
    ax[0].plot(xs, smooth(h["ld"]), color=COL[name], ls="--", label=f"{name}: потеря D")
    ax[1].plot(xs, smooth(h["d_real"]), color=COL[name], label=f"{name}: D на реальных")
    ax[1].plot(xs, smooth(h["d_fake"]), color=COL[name], ls="--", label=f"{name}: D на поддельных")
ax[0].axhline(2 * np.log(2), color="gray", lw=0.8, ls=":"); ax[0].text(20, 2 * np.log(2) + 0.04, "2 ln 2 ≈ 1,39", color="gray", fontsize=8)
ax[1].axhline(0.5, color="gray", lw=0.8, ls=":")
for a, t in zip(ax, ("Потери (скользящее среднее за 100 шагов)", "Средняя оценка D, сигмоида логита")):
    a.set_title(t, fontsize=10); a.set_xlabel("шаг"); a.legend(fontsize=7, ncol=2); a.grid(alpha=0.3)
    for s in (100, 300, 700, 1500):
        a.axvline(s, color="lightgray", lw=0.6)
fig.tight_layout(); fig.savefig(f"{F}/losses_seed101.png", dpi=150); plt.close(fig)

# 2. Журналы потерь всех 10 обучений варианта (seed 101…105)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for r in res["per_seed"]:
    h = r["hist"]; xs = (np.arange(len(h["lg"])) * 10)[9:]
    ax[0].plot(xs, smooth(h["lg"]), color=COL[r["mode"]], alpha=0.7, lw=1)
    ax[1].plot(xs, smooth(h["d_fake"]), color=COL[r["mode"]], alpha=0.7, lw=1)
ax[0].set_title("Потеря генератора, 5 seed (синий — A, оранжевый — B)", fontsize=10)
ax[1].set_title("D на поддельных, 5 seed (синий — A, оранжевый — B)", fontsize=10)
for a in ax:
    a.set_xlabel("шаг"); a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{F}/losses_all_seeds.png", dpi=150); plt.close(fig)

# 3. Снимки обучения под наблюдением: строки — A и B на шагах 100, 300, 700, 1500 (листы demo_gan.py, 16 образцов)
rows = [Image.open(f"{E}/demo/demo11_{m}_{s}.png").convert("RGB") for m in ("A", "B") for s in (100, 300, 700, 1500)]
w, h = rows[0].size; pad = 22
sheet = Image.new("RGB", (w, (h + pad) * len(rows)), "white")
from PIL import ImageDraw, ImageFont
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
except OSError:
    font = ImageFont.load_default()
for i, (im, (m, s)) in enumerate(zip(rows, [(m, s) for m in ("A", "B") for s in (100, 300, 700, 1500)])):
    d.text((4, i * (h + pad) + 3), f"режим {m}, шаг {s}", fill="black", font=font)
    sheet.paste(im, (0, i * (h + pad) + pad))
sheet.save(f"{F}/snapshots_A_B.png")
print("рисунки записаны в", F)

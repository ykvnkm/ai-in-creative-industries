# ЛР №8, вариант 2. Яковенко Максим Михайлович.
# Набор сцен 1 «диск» × фактор 2 «обычный автокодировщик (beta = 0) → вариационный (beta = 1)».
# Запуск в Engee из командной строки Julia (раздел 4 указаний):
#   PY = "/user/.user_venvs/py3.11/bin/python3"; run(`$PY lr08_variant2.py`)
# Файлы lab07_lib.py и lab08_lib.py лежат в той же папке; результаты пишутся в неё же.
# Шаг 2 (гипотеза и допуск до запуска) выполнен в журнале reports/journal.md.
import functools
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab07_lib as L
import lab08_lib as V

print = functools.partial(print, flush=True)
T_START = time.perf_counter()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Шаг 1. Среда и целостность библиотек
print("=" * 100)
print("Шаг 1. Среда")
print("дата и время (UTC):", datetime.now(timezone.utc).isoformat(timespec="seconds"))
print("python", platform.python_version(), "|", platform.platform())
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| устройство:", L.DEVICE, "| потоков torch:", torch.get_num_threads())
print("numpy", np.__version__)
LIBS = {}
for name, ref in (("lab07_lib.py", "d84b83a8"), ("lab08_lib.py", "f414c49b")):
    p = HERE / name
    LIBS[name] = dict(bytes=p.stat().st_size, sha256=sha256(p))
    print(f"{name}: {LIBS[name]['bytes']} байт, SHA-256 {LIBS[name]['sha256']}; совпадает с указаниями ({ref}…): {LIBS[name]['sha256'].startswith(ref)}")

# Шаг 3. Режимы A и B
VARIANT = 2
c, k = (VARIANT - 1) // 6, (VARIANT - 1) % 6
KIND, FACTOR = L.KINDS[c], V.FACTORS[k]
a_cfg, b_cfg = V.settings(FACTOR)
SEEDS = list(L.SEEDS)
print("=" * 100)
print(f"Шаг 3. Вариант {VARIANT}: набор {c + 1} «{KIND}», фактор {k + 1} «{FACTOR}»")
print("A:", a_cfg)
print("B:", b_cfg)
DIFF = {q: (a_cfg[q], b_cfg[q]) for q in a_cfg if a_cfg[q] != b_cfg[q]}
print("различаются поля:", DIFF)

Xtr, Xte, Ftr, Fte = L.dataset(KIND)
MEAN_PSNR = L.psnr_images(Xtr.mean(0, keepdim=True).expand_as(Xte), Xte)
print(f"данные: обучение {tuple(Xtr.shape)}, тест {tuple(Xte.shape)}; эталон «средняя картинка» {MEAN_PSNR:.3f} дБ")

# Обучение всех моделей серии (кэш get_model; variant_summary затем их переиспользует)
print("=" * 100)
print("Обучение: 2 режима × 5 seed, по 30 эпох")
TRAIN_SEC = {"A": [], "B": []}
for s in SEEDS:
    for mode, cfg in (("A", a_cfg), ("B", b_cfg)):
        t0 = time.perf_counter()
        V.get_model(KIND, cfg, s)
        dt = time.perf_counter() - t0
        TRAIN_SEC[mode].append(dt)
        print(f"  seed {s}, режим {mode} (beta = {cfg['beta']}): {dt:6.1f} с")
print(f"среднее время одного обучения: A {np.mean(TRAIN_SEC['A']):.1f} с, B {np.mean(TRAIN_SEC['B']):.1f} с")

# Показатели по каждому seed (схлопывание видно только здесь, среднее его скрывает)
PER_SEED = {"A": {}, "B": {}}
print("=" * 100)
print("Показатели по seed")
print(f"{'seed':>6} {'режим':>6} {'recon':>8} {'interp':>8} {'sample':>8} {'r2':>7}")
for s in SEEDS:
    for mode, cfg in (("A", a_cfg), ("B", b_cfg)):
        m = V.run(KIND, cfg, s)
        PER_SEED[mode][s] = m
        print(f"{s:>6} {mode:>6} {m['recon']:8.3f} {m['interp']:8.3f} {m['sample']:8.3f} {m['r2']:7.3f}")
COLLAPSED = [s for s in SEEDS if PER_SEED["B"][s]["recon"] < MEAN_PSNR + 1.0]
print("seed VAE с реконструкцией в пределах 1 дБ от эталона «средняя картинка»:", COLLAPSED or "нет")

# Шаг 4. Сводка варианта
print("=" * 100)
print("Шаг 4. V.variant_summary(2)")
result = V.variant_summary(VARIANT, seeds=SEEDS)
ROWS = {}
for name, a, s, b, d, q, sig in result["rows"]:
    ROWS[name] = dict(A=a, s=s, B=b, delta=d, ratio=q, sig2s=sig)
    print(f"{name:7s} A={a:7.3f} s={s:6.3f} B={b:7.3f} d={d:+7.3f} |d|/s={q:7.2f} 2s={'да' if sig else 'нет'}")

# Шаг 7. Классификация по допуску, записанному до запуска
TOL = dict(recon=1.0, interp=1.0, sample=1.0, r2=0.05)


def classify(d, t):
    return "улучшение" if d > t else "ухудшение" if d < -t else "нейтрально"


print("=" * 100)
print("Шаг 7. Классификация по допуску (PSNR ±1,0 дБ; R² ±0,05)")
CLASSES = {}
for name, r in ROWS.items():
    CLASSES[name] = classify(r["delta"], TOL[name])
    agree = (CLASSES[name] != "нейтрально") == r["sig2s"]
    print(f"{name:7s} Δ = {r['delta']:+.3f} → {CLASSES[name]:10s} | 2s: {'значимо' if r['sig2s'] else 'не значимо':10s} | "
          f"{'согласуются' if agree else 'РАСХОДЯТСЯ'}")
print("чувствительность к допуску (множитель × основной допуск):")
SENS = {}
for f in (0.25, 0.5, 1.0, 1.5, 2.0, 4.0):
    SENS[f] = {name: classify(r["delta"], TOL[name] * f) for name, r in ROWS.items()}
    print(f"  × {f:<4}: " + "; ".join(f"{n} {c_}" for n, c_ in SENS[f].items()))

# Шаг 5. Диагностика: разброс mu по осям, норма кодов, KL по осям и активные оси
print("=" * 100)
print("Шаг 5. Диагностика латентного пространства (тестовая выборка)")
DIAG = {"A": {}, "B": {}}
for mode, cfg in (("A", a_cfg), ("B", b_cfg)):
    for s in SEEDS:
        model = V.get_model(KIND, cfg, s)
        with torch.no_grad():
            mu, lv = model.encode_dist(Xte)
            kl_axis = (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean(0)
        sd = mu.std(0).cpu().numpy()
        row = dict(mu_std=[round(float(v), 3) for v in sd], mu_mean_abs=float(mu.mean(0).abs().max()), norm=float(mu.norm(dim=1).mean()))
        if cfg["beta"] > 0:
            row["kl_axis"] = [round(float(v), 3) for v in kl_axis.cpu()]
            row["active"] = int((kl_axis > 0.05).sum())
            row["sigma_mean"] = float(torch.exp(0.5 * lv).mean())
        DIAG[mode][s] = row
        extra = f"; KL по осям {row['kl_axis']}; активных осей {row['active']}; среднее σ {row['sigma_mean']:.3f}" if cfg["beta"] > 0 else ""
        print(f"{mode} seed {s}: разброс mu {row['mu_std']}; max|среднее mu| {row['mu_mean_abs']:.2f}; ‖mu‖ {row['norm']:.2f}{extra}")
print(f"для сравнения: норма сэмпла из N(0, I) при d = 8 в среднем {float(torch.randn(100000, 8, generator=torch.Generator().manual_seed(0)).norm(dim=1).mean()):.2f}")
for mode in ("A", "B"):
    lo = min(min(DIAG[mode][s]["mu_std"]) for s in SEEDS); hi = max(max(DIAG[mode][s]["mu_std"]) for s in SEEDS)
    print(f"режим {mode}: разброс mu по осям {lo:.2f}–{hi:.2f}; средняя ‖mu‖ {np.mean([DIAG[mode][s]['norm'] for s in SEEDS]):.2f}"
          + (f"; активных осей {[DIAG[mode][s]['active'] for s in SEEDS]}" if mode == "B" else ""))


# Шаг 8. Сэмплы обычного автокодировщика: из N(0, I) и из распределения mu, оценённого на обучающих данных
@torch.no_grad()
def sample_psnr(model, z):
    """Тот же показатель, что в V.metrics: средний PSNR ближайшей тестовой сцены."""
    xs = model.decode(z).flatten(1)
    mse = torch.cdist(xs, Xte.flatten(1)).pow(2) / xs.shape[1]
    return float((10 * torch.log10(1.0 / mse.min(1).values.clamp_min(1e-10))).mean())


@torch.no_grad()
def fitted_gauss(model):
    mu = model.encode(Xtr).double()
    m = mu.mean(0)
    cov = torch.cov(mu.T) + 1e-6 * torch.eye(mu.shape[1], dtype=mu.dtype)
    return m, torch.linalg.cholesky(cov)


print("=" * 100)
print("Шаг 8. Сэмплы: N(0, I) и N(mu_train, Sigma_train) у обычного автокодировщика, N(0, I) у вариационного")
SAMPLES = {"AE_N01": [], "AE_fitted": [], "VAE_N01": []}
for s in SEEDS:
    ae, vae = V.get_model(KIND, a_cfg, s), V.get_model(KIND, b_cfg, s)
    g = torch.Generator(device="cpu").manual_seed(1000 + s)          # те же случайные векторы, что в V.metrics
    eps = torch.randn((V.N_SAMPLES, 8), generator=g)
    m, Lc = fitted_gauss(ae)
    z_fit = (m + eps.double() @ Lc.T).float()
    SAMPLES["AE_N01"].append(sample_psnr(ae, eps.to(L.DEVICE)))
    SAMPLES["AE_fitted"].append(sample_psnr(ae, z_fit.to(L.DEVICE)))
    SAMPLES["VAE_N01"].append(sample_psnr(vae, eps.to(L.DEVICE)))
    print(f"seed {s}: AE N(0, I) {SAMPLES['AE_N01'][-1]:.2f} дБ | AE N(mu, Sigma) {SAMPLES['AE_fitted'][-1]:.2f} дБ | VAE N(0, I) {SAMPLES['VAE_N01'][-1]:.2f} дБ")
for q, v in SAMPLES.items():
    print(f"{q:10s} среднее {np.mean(v):.2f} дБ, s {np.std(v, ddof=1):.2f}")

# Форма дисков на пути интерполяции: связные компоненты маски, центр и радиус, полоса промежуточных значений («призраки»)
try:
    from scipy.ndimage import label as _label

    def n_components(mask):
        return int(_label(mask)[1])
except ImportError:
    def n_components(mask):
        seen, n = np.zeros_like(mask, dtype=bool), 0
        for i0, j0 in zip(*np.nonzero(mask)):
            if seen[i0, j0]:
                continue
            n += 1; stack = [(i0, j0)]; seen[i0, j0] = True
            while stack:
                i, j = stack.pop()
                for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not seen[a, b]:
                        seen[a, b] = True; stack.append((a, b))
        return n

_YY, _XX = np.mgrid[0:L.N, 0:L.N].astype(np.float64)


def shape_stats(imgs, ref_geo=None, thr=0.675, band=(0.45, 0.85)):
    red = imgs[:, 0].cpu().numpy().astype(np.float64)
    masks = red > thr
    area = masks.sum((1, 2))
    a = np.maximum(area, 1)
    cx, cy = (masks * _XX).sum((1, 2)) / a, (masks * _YY).sum((1, 2)) / a
    geo = dict(cx=cx, cy=cy, r=np.sqrt(area / np.pi))
    out = dict(one_component=float(np.mean([n_components(m) == 1 for m in masks])),
               ghost_band=float(((red > band[0]) & (red < band[1])).sum((1, 2)).mean()),
               empty=int((area == 0).sum()))
    if ref_geo is not None:
        for q in ("cx", "cy", "r"):
            out[q + "_mae"] = float(np.abs(geo[q] - ref_geo[q]).mean())
    return out, geo


print("=" * 100)
print("Форма дисков на пути интерполяции: 200 пар × t = 0,25; 0,5; 0,75; все 5 seed")
ia, ib, tg = V.interp_targets(KIND)
SHAPE = {}
target_all = torch.cat([tg[t] for t in (0.25, 0.5, 0.75)])
SHAPE["true"], ref_geo = shape_stats(target_all)
blend = torch.cat([(1 - t) * Xte[ia] + t * Xte[ib] for t in (0.25, 0.5, 0.75)])
SHAPE["pixel_blend"], _ = shape_stats(blend, ref_geo)
# Эталонные сцены рисуются без сглаживания (полоса у них 0), поэтому полоса пути сравнивается с полосой
# реконструкций тестовых сцен той же моделью: размытие края декодировщиком есть и там, «призраков» нет.
test_geo = shape_stats(Xte)[1]
for mode, cfg in (("A", a_cfg), ("B", b_cfg)):
    per, per_rec = [], []
    for s in SEEDS:
        model = V.get_model(KIND, cfg, s)
        with torch.no_grad():
            mu = model.encode(Xte)
            dec = torch.cat([model.decode((1 - t) * mu[ia] + t * mu[ib]) for t in (0.25, 0.5, 0.75)])
            rec = model.decode(mu)
        per.append(shape_stats(dec, ref_geo)[0])
        per_rec.append(shape_stats(rec, test_geo)[0])
    SHAPE[mode] = {q: float(np.mean([p[q] for p in per])) for q in per[0]}
    SHAPE[mode]["one_component_min_seed"] = float(min(p["one_component"] for p in per))
    SHAPE["recon_" + mode] = {q: float(np.mean([p[q] for p in per_rec])) for q in per_rec[0]}
for name in ("true", "pixel_blend", "recon_A", "A", "recon_B", "B"):
    r = SHAPE[name]
    geo = "" if name == "true" else f"; ошибка центра x {r['cx_mae']:.2f}, y {r['cy_mae']:.2f}, радиуса {r['r_mae']:.2f} пикс."
    print(f"{name:12s} одна компонента {100 * r['one_component']:5.1f} %; полоса 0,45–0,85: {r['ghost_band']:6.1f} пикс.; пустых масок {r['empty']}{geo}")
CRIT = {}
for name, ref in (("pixel_blend", "true"), ("A", "recon_A"), ("B", "recon_B")):
    r = SHAPE[name]
    CRIT[name] = bool(r["one_component"] >= 0.95 and max(r["cx_mae"], r["cy_mae"], r["r_mae"]) < 1.0
                      and r["ghost_band"] <= 1.5 * SHAPE[ref]["ghost_band"])
    print(f"критерий для {name} (полоса относительно {ref}): {'выполнен' if CRIT[name] else 'НЕ выполнен'}")

# Шаг 6. Лист сравнения comparison.png (модели seed 101; 8 столбцов, увеличение ×4)
from PIL import Image


def strip(imgs):
    return np.concatenate(list(imgs.permute(0, 2, 3, 1).cpu().numpy()), axis=1)


ae, vae = V.get_model(KIND, a_cfg, 101), V.get_model(KIND, b_cfg, 101)
g = torch.Generator(device="cpu").manual_seed(5)
z = torch.randn((8, 8), generator=g)
m, Lc = fitted_gauss(ae)
t8 = torch.linspace(0, 1, 8).unsqueeze(1)
f0, f2 = Fte[0].cpu().numpy(), Fte[2].cpu().numpy()
rows = []
with torch.no_grad():
    rows.append(strip(ae.decode(z.to(L.DEVICE))))                                        # 1: AE, z ~ N(0, I)
    rows.append(strip(ae.decode((m + z.double() @ Lc.T).float().to(L.DEVICE))))           # 2: AE, z ~ N(mu, Sigma)
    rows.append(strip(vae.decode(z.to(L.DEVICE))))                                       # 3: VAE, z ~ N(0, I)
    rows.append(strip(torch.tensor(np.stack([L.render(KIND, (1 - t) * f0 + t * f2) for t in t8[:, 0].numpy()]))))  # 4: истинный путь
    for model in (ae, vae):                                                               # 5, 6: линейный путь в латентном пространстве
        za, zb = model.encode(Xte[0:1]), model.encode(Xte[2:3])
        rows.append(strip(model.decode((1 - t8.to(L.DEVICE)) * za + t8.to(L.DEVICE) * zb)))
    rows.append(strip((1 - t8[:, :, None, None]) * Xte[0:1].cpu() + t8[:, :, None, None] * Xte[2:3].cpu()))  # 7: смешивание пикселей
sheet = Image.fromarray((np.concatenate(rows, axis=0) * 255).round().astype("uint8")).resize((8 * 32 * 4, len(rows) * 32 * 4), Image.NEAREST)
sheet.save(HERE / "comparison.png")
print("=" * 100)
print("Шаг 6. comparison.png сохранён; строки: 1 AE N(0,I); 2 AE N(mu,Sigma); 3 VAE N(0,I); 4 истинный путь; 5 AE путь; 6 VAE путь; 7 смешивание пикселей")

# Сверка режима B с режимом A демонстрационного варианта (та же конфигурация, таблица 6.3 указаний)
DEMO_A = dict(recon=(26.37, 0.61), interp=(22.55, 0.48), sample=(23.24, 0.56), r2=(0.899, 0.022))
print("=" * 100)
print("Сверка: мой режим B (VAE, beta = 1) и режим A демонстрационного варианта")
REPRO = {}
for name, (ref, s_ref) in DEMO_A.items():
    d = ROWS[name]["B"] - ref
    REPRO[name] = dict(mine=ROWS[name]["B"], demo=ref, diff=d, diff_in_s=d / s_ref)
    print(f"{name:7s} мой {ROWS[name]['B']:7.3f} | указания {ref:7.3f} | разность {d:+.3f} ({d / s_ref:+.2f} s указаний)")

# Сохранение
T_TOTAL = time.perf_counter() - T_START
results = dict(
    variant=VARIANT, kind=KIND, factor=FACTOR, A=a_cfg, B=b_cfg, diff=DIFF, seeds=SEEDS,
    environment=dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__, numpy=np.__version__,
                     device=L.DEVICE, threads=torch.get_num_threads(), date_utc=datetime.now(timezone.utc).isoformat(timespec="seconds")),
    libraries=LIBS, mean_image_psnr=MEAN_PSNR, train_seconds=TRAIN_SEC,
    per_seed={mode: {str(s): v for s, v in d.items()} for mode, d in PER_SEED.items()}, collapsed_vae_seeds=COLLAPSED,
    summary=ROWS, tolerance=TOL, classes=CLASSES, sensitivity={str(f): v for f, v in SENS.items()},
    diagnostics={mode: {str(s): v for s, v in d.items()} for mode, d in DIAG.items()},
    samples=SAMPLES, interpolation_shape=SHAPE, interpolation_shape_criterion=CRIT, repro_vs_demo=REPRO, total_seconds=T_TOTAL,
)
(HERE / "results_v02.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print("=" * 100)
print(f"всего {T_TOTAL / 60:.1f} мин")
for f in ("results_v02.json", "comparison.png"):
    print(f"{f}: SHA-256 {sha256(HERE / f)}")

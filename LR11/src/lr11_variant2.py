# ЛР 11, вариант 2: набор «диск» × ширина генератора 1 → 0,5. Яковенко Максим Михайлович.
# Запуск в Engee из папки, где лежат lab07_lib.py и lab11_lib.py:  LAB_THREADS=1 python lr11_variant2.py
# Сводку считает variant_summary(2) из библиотеки без изменений. Функция run() обёрнута регистратором: он записывает
# метрики и журнал потерь каждого обучения, число параметров генератора и для seed 101 сохраняет лист из 32 образцов.
# Результаты: results_v02.json, samples/<режим>_s101.png, печать в stdout.
import json, os, platform, resource, sys, time
os.environ.setdefault("LAB_THREADS", "1")   # как в прогоне преподавателя (run_lab.py) — для сверки значений по seed
import numpy as np
import torch
from PIL import Image
import lab11_lib as L

N = 2
KIND, A_CFG, B_CFG = L.variant_settings(N)
os.makedirs("samples", exist_ok=True)
TEACHER = "teacher/results_30_summary.txt"   # сводка прогона преподавателя, выдана с указаниями


def cpu_model():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return platform.processor()


def sheet(G, path, n=32, cols=8, scale=4):
    """Лист n образцов (seed выборки 7, как в demo_gan.py) сеткой cols × n/cols, увеличение без сглаживания."""
    x = (L.sample(G, n, 7).permute(0, 2, 3, 1).numpy() * 255).astype("uint8")
    rows = [np.concatenate(list(x[r * cols:(r + 1) * cols]), axis=1) for r in range(n // cols)]
    im = Image.fromarray(np.concatenate(rows, axis=0))
    im.resize((im.width * scale, im.height * scale), Image.NEAREST).save(path)


rec = []
_run = L.run


def run_logged(kind, cfg, seed):
    m, G, hist = _run(kind, cfg, seed)
    mode = "A" if cfg == A_CFG else "B"
    rec.append(dict(mode=mode, seed=seed, **m, params_g=sum(p.numel() for p in G.parameters()), hist=hist))
    if seed == 101:
        sheet(G, f"samples/{mode}_s101.png")
    print(f"  {mode} seed {seed}: nn_psnr {m['nn_psnr']:.2f}  coverage {m['coverage']:.3f}  diversity {m['diversity']:.3f}  "
          f"{m['seconds']:.1f} с  (ld {np.mean(hist['ld'][-10:]):.3f}, lg {np.mean(hist['lg'][-10:]):.3f})", flush=True)
    return m, G, hist


L.run = run_logged

env = dict(date_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), python=sys.version.split()[0], torch=torch.__version__,
           numpy=np.__version__, cuda=torch.cuda.is_available(), torch_threads=torch.get_num_threads(),
           lab_threads=os.environ["LAB_THREADS"], cpu=cpu_model(), cpu_count=os.cpu_count())
print("Среда:", json.dumps(env, ensure_ascii=False))
print(f"Вариант {N}: набор {KIND}, фактор {L.FACTORS[(N - 1) % 6]}")
print("A:", A_CFG)
print("B:", B_CFG)
print("отличие A и B:", {k: (A_CFG[k], B_CFG[k]) for k in A_CFG if A_CFG[k] != B_CFG[k]}, flush=True)

t0 = time.perf_counter()
M = L.variant_summary(N)
t_all = time.perf_counter() - t0
print(f"variant_summary: {t_all / 60:.1f} мин (10 обучений)")

print("\nСводка variant_summary (как в указаниях):")
print(f"{'метрика':<10}{'A':>9}{'s':>8}{'B':>9}{'Δ':>9}{'|Δ|/s':>7}  |Δ|>2s")
for m, a, s, b, d, q, f in M["rows"]:
    nd = 2 if m in ("nn_psnr", "seconds") else 3
    print(f"{m:<10}{a:>9.{nd}f}{s:>8.{nd}f}{b:>9.{nd}f}{d:>+9.{nd}f}{q:>7.1f}  {'да' if f else 'нет'}")

A = sorted((r for r in rec if r["mode"] == "A"), key=lambda r: r["seed"])
B = sorted((r for r in rec if r["mode"] == "B"), key=lambda r: r["seed"])
check = L.summarize({m: [r[m] for r in A] for m in L.METRICS}, {m: [r[m] for r in B] for m in L.METRICS})
same = all(x == y or abs(x - y) < 1e-12 for r1, r2 in zip(M["rows"], check) for x, y in zip(r1[1:6], r2[1:6]))
print("\nПересчёт сводки по записанным значениям совпал с variant_summary:", same)
print("параметров генератора: A", A[0]["params_g"], " B", B[0]["params_g"])

print("\nПо seed:")
for m in ("nn_psnr", "coverage", "diversity", "seconds"):
    nd = 2 if m in ("nn_psnr", "seconds") else 3
    print(f"{m:<10} A " + " ".join(f"{r[m]:.{nd}f}" for r in A) + " | B " + " ".join(f"{r[m]:.{nd}f}" for r in B)
          + " | B−A " + " ".join(f"{b[m] - a[m]:+.{nd}f}" for a, b in zip(A, B)))
print("потери за последние 100 шагов (ld / lg / D(real) / D(fake)):")
for r in A + B:
    h = r["hist"]
    print(f"  {r['mode']} seed {r['seed']}: " + " / ".join(f"{np.mean(h[k][-10:]):.3f}" for k in ("ld", "lg", "d_real", "d_fake")))

teacher = {}
if os.path.exists(TEACHER):
    lines = open(TEACHER, encoding="utf-8-sig").read().splitlines()
    v1 = dict(p.split("=") for p in lines[0].split()[1:])                  # режим A на наборе «диск» общий для вариантов 1–6
    row = next(l for l in lines if l.startswith(f"{N}|"))
    teacher = dict(nnA=[float(x) for x in v1["nnA"].split(",")], cvA=[float(x) for x in v1["cvA"].split(",")], row=row)
    print("\nСверка с прогоном преподавателя (results_30_summary.txt):")
    mine_nn = [round(r["nn_psnr"], 3) for r in A]; mine_cv = [round(r["coverage"], 3) for r in A]
    print(f"  nn_psnr A по seed: мой {mine_nn}  преподаватель {teacher['nnA']}  совпало: {mine_nn == teacher['nnA']}")
    print(f"  coverage A по seed: мой {mine_cv}  преподаватель {teacher['cvA']}  совпало: {mine_cv == teacher['cvA']}")
    print("  строка варианта 2 у преподавателя:", row)
    short = dict(nn_psnr="nn", coverage="cv", diversity="dv", seconds="sec")   # формат строк summarize_lab.py
    nd = lambda m: 2 if m in ("nn_psnr", "seconds") else 3
    mine = " ".join(f"{short[m]}:{a:.{nd(m)}f}/{s:.{nd(m)}f}/{b:.{nd(m)}f}/{d:+.{nd(m)}f}/{q:.1f}/{int(f)}" for m, a, s, b, d, q, f in M["rows"])
    print("  моя строка варианта 2:            ", f"{N}|1|{L.FACTORS[(N - 1) % 6]}|{mine}")

peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(f"\nпик памяти процесса: {peak_gb:.2f} ГБ")
json.dump(dict(variant=N, kind=KIND, factor=L.FACTORS[(N - 1) % 6], cfg_a=A_CFG, cfg_b=B_CFG, env=env, minutes=t_all / 60,
               peak_rss_gb=peak_gb, rows=M["rows"], rows_recomputed_match=same, per_seed=rec, teacher=teacher),
          open("results_v02.json", "w", encoding="utf-8"), ensure_ascii=False)
print("записано: results_v02.json, samples/A_s101.png, samples/B_s101.png")

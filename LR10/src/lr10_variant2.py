# ЛР 10, вариант 2. Яковенко Максим Михайлович.
# Запуск в Engee из папки, где лежит lab10_lib.py:  python lr10_variant2.py
# Сводку считает variant_summary(2, save_dir="v02") из библиотеки без изменений, листы слепого сравнения — blind_sheets(2, "v02").
# Вызов measure() обёрнут регистратором: он записывает значения по seed и сохраняет изображения ещё и в PNG (без потерь).
# Ключ слепого сравнения (v02/blind_key.json) скрипт не читает и не печатает.
import hashlib, json, os, platform, resource, sys, time
import numpy as np
import torch, diffusers, transformers, scipy, PIL, huggingface_hub
import lab10_lib as L

N = 2
C, K = (N - 1) // 6, (N - 1) % 6
OUT = "v02"
os.makedirs(OUT, exist_ok=True)
# Демонстрационный вариант 1 (таблицы 6.3 указаний): модель A та же — SD Turbo на брифе 1
DEMO_A = dict(clip_brief=(25.17, 0.80), clip_text=(12.40, 0.94), edge=(0.0527, 0.0039), contrast=(0.1750, 0.0171))
DEMO_A_BRIEF = [24.46, 26.13, 24.21, 25.53, 25.50]


def sha(img):
    return hashlib.sha256(np.asarray(img).tobytes()).hexdigest()


def cpu_model():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return platform.processor()


rec = []
_measure = L.measure


def measure_logged(pipe, clip, key, ctx, seed, cfg=None):
    r, img = _measure(pipe, clip, key, ctx, seed, cfg)
    img.save(os.path.join(OUT, f"{key}_s{seed}.png"))
    rec.append(dict(model=key, seed=seed, **r, pixels_sha256=sha(img)))
    print(f"  {key} seed {seed}: clip_brief {r['clip_brief']:.2f}  clip_text {r['clip_text']:.2f}  "
          f"edge {r['edge']:.4f}  contrast {r['contrast']:.4f}  {r['seconds']:.1f} с", flush=True)
    return r, img


L.measure = measure_logged

_load = L.load_model
load_seconds = {}


def load_logged(key):
    t0 = time.perf_counter(); p = _load(key); load_seconds[key] = time.perf_counter() - t0
    print(f"загрузка {key}: {load_seconds[key]:.1f} с; планировщик {type(p.scheduler).__name__}", flush=True)
    return p


L.load_model = load_logged

a_key, b_key = L.PAIRS[K]
env = dict(date_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), python=sys.version.split()[0],
           torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
           numpy=np.__version__, scipy=scipy.__version__, pillow=PIL.__version__, huggingface_hub=huggingface_hub.__version__,
           cuda=torch.cuda.is_available(), torch_threads=torch.get_num_threads(), cpu=cpu_model(), cpu_count=os.cpu_count(),
           clip=L.CLIP_ID, clip_revision=L.CLIP_REVISION, seeds=list(L.SEEDS),
           models={k: L.MODELS[k] for k in (a_key, b_key)})
print("Среда:", json.dumps(env, ensure_ascii=False), flush=True)
print(f"Вариант {N}: бриф {C + 1}, пара {K + 1}: A = {a_key}, B = {b_key}")
print("бриф:", repr(L.BRIEFS[C]), flush=True)

t0 = time.perf_counter()
M = L.variant_summary(N, save_dir=OUT)
t_all = time.perf_counter() - t0
print(f"variant_summary: {t_all:.1f} с (CLIP, две загрузки, два прогрева, 10 изображений)", flush=True)
L.blind_sheets(N, OUT)   # ключ не печатается
print("листы слепого сравнения:", sorted(f for f in os.listdir(OUT) if f.startswith("blind_") and f.endswith(".png")))

print("\nСводка variant_summary (как в указаниях):")
print(f"{'метрика':<12}{'A':>10}{'s':>9}{'B':>10}{'Δ':>10}{'|Δ|/s':>8}  |Δ|>2s")
for m, a, s, b, d, q, f in M["rows"]:
    nd = 4 if m in ("edge", "contrast") else 2
    print(f"{m:<12}{a:>10.{nd}f}{s:>9.{nd}f}{b:>10.{nd}f}{d:>+10.{nd}f}{q:>8.1f}  {'да' if f else 'нет'}")

A = [r for r in rec if r["model"] == a_key]; B = [r for r in rec if r["model"] == b_key]
rows_check = L.compare(A, B)
same = all(x == y or abs(x - y) < 1e-12 for r1, r2 in zip(M["rows"], rows_check) for x, y in zip(r1[1:6], r2[1:6]))
print("\nПересчёт сводки по записанным значениям совпал с variant_summary:", same)
ratio = float(np.mean([r["seconds"] for r in B]) / np.mean([r["seconds"] for r in A]))
print(f"отношение времён B/A: {ratio:.2f}")

print("\nПо seed (clip_brief):")
ba = [r["clip_brief"] for r in A]; bb = [r["clip_brief"] for r in B]
print("seed  " + "  ".join(f"{r['seed']:>7}" for r in A))
print("A     " + "  ".join(f"{x:>7.2f}" for x in ba))
print("B     " + "  ".join(f"{x:>7.2f}" for x in bb))
dd = [b - a for a, b in zip(ba, bb)]
print("B − A " + "  ".join(f"{x:>+7.2f}" for x in dd))

print("\nСверка модели A с демонстрационным вариантом 1 (таблицы 6.3):")
for m in DEMO_A:
    vals = [r[m] for r in A]
    nd = 4 if m in ("edge", "contrast") else 2
    mine = (round(float(np.mean(vals)), nd), round(float(np.std(vals, ddof=1)), nd))
    print(f"  {m:<11} мой {mine[0]:.{nd}f} (s {mine[1]:.{nd}f})  демо {DEMO_A[m][0]:.{nd}f} (s {DEMO_A[m][1]:.{nd}f})  совпало: {mine == DEMO_A[m]}")
print(f"  clip_brief по seed: мой {[round(x, 2) for x in ba]}  демо {DEMO_A_BRIEF}  совпало: {[round(x, 2) for x in ba] == DEMO_A_BRIEF}")

peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(f"\nпик памяти процесса: {peak_gb:.1f} ГБ")
json.dump(dict(variant=N, ctx=C + 1, pair=[a_key, b_key], brief=L.BRIEFS[C], env=env, load_seconds=load_seconds,
               variant_summary_seconds=t_all, peak_rss_gb=peak_gb, rows=M["rows"], rows_recomputed_match=same,
               time_ratio=ratio, per_seed=rec),
          open("results_v02.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("записано: results_v02.json, изображения и листы слепого сравнения в", OUT)

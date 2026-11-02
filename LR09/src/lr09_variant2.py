# ЛР 9, вариант 2, часть 1. Яковенко Максим Михайлович.
# Запуск в Engee из папки, где лежит lab09_lib.py:  python lr09_variant2.py
# Сводку считает variant_summary() из библиотеки без изменений. Вызов measure() обёрнут регистратором: он записывает
# значения по seed (сходство со всеми целями контекста, edge, contrast, время) и сохраняет изображения в PNG.
# Результаты: v02/A_s<seed>.png, v02/B_s<seed>.png, results_v02.json, печать в stdout.
import hashlib, io, json, os, platform, resource, sys, time
import numpy as np
import torch, diffusers, transformers, scipy, PIL, huggingface_hub
import lab09_lib as L

N = 2
C, K = (N - 1) // 6, (N - 1) % 6
OUT = "v02"
os.makedirs(OUT, exist_ok=True)
# Демонстрационный вариант 1 (таблицы 6.3 указаний): режим A тот же, что у варианта 2
DEMO_A = dict(clip_subject=(25.24, 1.78), edge=(0.0375, 0.0025), contrast=(0.1808, 0.0056))
DEMO_A_FLAT = [14.80, 10.40, 12.77, 14.37, 8.16]   # сходство режима A с «flat vector illustration», seed 101…105


def png_bytes(img):
    b = io.BytesIO(); img.save(b, format="PNG"); return b.getvalue()


def sha(img):
    return hashlib.sha256(np.asarray(img).tobytes()).hexdigest()


def cpu_model():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return platform.processor()


rec = []
_measure = L.measure


def measure_logged(pipe, clip, c, prompt, seed):
    r, img = _measure(pipe, clip, c, prompt, seed)
    mode = "A" if prompt == L.prompt_a(C) else "B"
    img.save(os.path.join(OUT, f"{mode}_s{seed}.png"))
    rec.append(dict(mode=mode, seed=seed, prompt=prompt, sims=r["sims"], edge=r["edge"], contrast=r["contrast"],
                    seconds=r["seconds"], pixels_sha256=sha(img)))
    print(f"  {mode} seed {seed}: clip_target {r['sims'][1 + K]:.2f}  clip_subject {r['sims'][0]:.2f}  "
          f"edge {r['edge']:.4f}  contrast {r['contrast']:.4f}  {r['seconds']:.1f} с", flush=True)
    return r, img


L.measure = measure_logged

env = dict(date_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), python=sys.version.split()[0],
           torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
           numpy=np.__version__, scipy=scipy.__version__, pillow=PIL.__version__, huggingface_hub=huggingface_hub.__version__,
           cuda=torch.cuda.is_available(), torch_threads=torch.get_num_threads(), cpu=cpu_model(), cpu_count=os.cpu_count(),
           model=L.MODEL_ID, revision=L.REVISION, clip=L.CLIP_ID, clip_revision=L.CLIP_REVISION,
           steps=L.STEPS, guidance=L.GUIDANCE, size=L.SIZE, seeds=list(L.SEEDS))
print("Среда:", json.dumps(env, ensure_ascii=False), flush=True)
print(f"Вариант {N}: контекст {C + 1}, элемент {K + 1} ({L.FACTORS[K]})")
print("A:", repr(L.prompt_a(C)))
print("B:", repr(L.prompt_b(C, K)))
print("текст-цель CLIP:", repr(L.ELEMENTS[C][K][1]), flush=True)

t0 = time.perf_counter(); pipe = L.load_pipe(); t_pipe = time.perf_counter() - t0
t0 = time.perf_counter(); clip = L.load_clip(); t_clip = time.perf_counter() - t0
print(f"загрузка: SD Turbo {t_pipe:.1f} с, CLIP {t_clip:.1f} с", flush=True)

t0 = time.perf_counter()
M = L.variant_summary(pipe, clip, N)
t_series = time.perf_counter() - t0
print(f"серия (прогрев + 10 изображений + CLIP): {t_series:.1f} с", flush=True)

print("\nСводка variant_summary (как в указаниях):")
print(f"{'метрика':<13}{'A':>10}{'s':>9}{'B':>10}{'Δ':>10}{'|Δ|/s':>8}  |Δ|>2s")
for m, a, s, b, d, q, f in M["rows"]:
    nd = 4 if m in ("edge", "contrast") else 2
    print(f"{m:<13}{a:>10.{nd}f}{s:>9.{nd}f}{b:>10.{nd}f}{d:>+10.{nd}f}{q:>8.1f}  {'да' if f else 'нет'}")

A = [r for r in rec if r["mode"] == "A"]; B = [r for r in rec if r["mode"] == "B"]
rows_check = L.compare([dict(sims=r["sims"], edge=r["edge"], contrast=r["contrast"]) for r in A],
                       [dict(sims=r["sims"], edge=r["edge"], contrast=r["contrast"]) for r in B], K)
same = all(x == y or abs(x - y) < 1e-12 for r1, r2 in zip(M["rows"], rows_check) for x, y in zip(r1[1:6], r2[1:6]))
print("\nПересчёт сводки по записанным значениям совпал с variant_summary:", same)

print("\nПо seed (clip_target):")
ta = [r["sims"][1 + K] for r in A]; tb = [r["sims"][1 + K] for r in B]
print("seed  " + "  ".join(f"{r['seed']:>7}" for r in A))
print("A     " + "  ".join(f"{x:>7.2f}" for x in ta))
print("B     " + "  ".join(f"{x:>7.2f}" for x in tb))
dd = [b - a for a, b in zip(ta, tb)]
print("B − A " + "  ".join(f"{x:>+7.2f}" for x in dd))
md, sd = float(np.mean(dd)), float(np.std(dd, ddof=1))
print(f"парные разности: положительных {sum(x > 0 for x in dd)} из 5; среднее {md:+.2f}, s разностей {sd:.2f}, среднее/s {md / sd:.2f}")

print("\nСверка режима A с демонстрационным вариантом 1 (таблицы 6.3):")
for m, i in (("clip_subject", None), ("edge", None), ("contrast", None)):
    vals = [r["sims"][0] if m == "clip_subject" else r[m] for r in A]
    nd = 2 if m == "clip_subject" else 4
    mine = (round(float(np.mean(vals)), nd), round(float(np.std(vals, ddof=1)), nd))
    print(f"  {m:<13} мой {mine[0]:.{nd}f} (s {mine[1]:.{nd}f})  демо {DEMO_A[m][0]:.{nd}f} (s {DEMO_A[m][1]:.{nd}f})  совпало: {mine == DEMO_A[m]}")
flat = [round(r["sims"][1], 2) for r in A]
print(f"  flat vector по seed: мой {flat}  демо {DEMO_A_FLAT}  совпало: {flat == DEMO_A_FLAT}")

print("\nОпределённость: повторная генерация seed 101", flush=True)
rep = {}
for mode, prompt in (("A", L.prompt_a(C)), ("B", L.prompt_b(C, K))):
    img = L.generate(pipe, prompt, 101)
    first = next(r for r in rec if r["mode"] == mode and r["seed"] == 101)
    rep[mode] = sha(img) == first["pixels_sha256"]
    print(f"  {mode}: пиксели совпали с серией бит в бит — {rep[mode]}")

peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
print(f"\nпик памяти процесса: {peak_gb:.1f} ГБ; среднее время изображения A {np.mean([r['seconds'] for r in A]):.1f} с, "
      f"B {np.mean([r['seconds'] for r in B]):.1f} с")

json.dump(dict(variant=N, ctx=C + 1, factor=L.FACTORS[K], prompt_a=L.prompt_a(C), prompt_b=L.prompt_b(C, K),
               target=L.ELEMENTS[C][K][1], texts=L.texts_for(C), env=env, load_seconds=dict(pipe=t_pipe, clip=t_clip),
               series_seconds=t_series, peak_rss_gb=peak_gb, rows=M["rows"], rows_recomputed_match=same,
               per_seed=rec, paired=dict(diffs=dd, mean=md, sd=sd), repeat_seed101_bit_identical=rep),
          open("results_v02.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("записано: results_v02.json, изображения в", OUT)

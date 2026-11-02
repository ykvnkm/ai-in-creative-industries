# ЛР 9, вариант 2, часть 2: оценка девяти ответов языковой модели функциями lab09_llm.py (без изменений).
# Запуск из корня работы LR09:  python src/lr09_part2_score.py
# Результат: печать таблицы K1–K4, reports/k1k4.json (K1–K3 по каждому ответу, ошибки схемы, K4 по приёмам).
import hashlib, json, os, sys

sys.path.insert(0, os.path.join("assignment", "src"))
import lab09_llm as L

CTX = 1
out = {}
for a in ("zero", "few", "step"):
    paths = [os.path.join("answers", f"ctx{CTX}_{a}_r{i}.txt") for i in (1, 2, 3)]
    rs = [L.score_file(p, CTX) for p in paths]
    out[a] = dict(k1=sum(r["k1"] for r in rs) / 3, k2=sum(r["k2"] for r in rs) / 3, k3=sum(r["k3"] for r in rs) / 3,
                  k4=L.stability(paths), each=[dict(file=p, sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(), **r)
                                                for p, r in zip(paths, rs)])

# K4 по полям отдельно и выбор style/composition по объектам — для разбора, откуда берётся нестабильность
for a in out:
    arrs = [L.extract_array(open(e["file"], encoding="utf-8").read())[0] for e in out[a]["each"]]
    out[a]["choices"] = {f: [[o.get(f) for o in arr] for arr in arrs] for f in ("style", "composition")}
    for f in ("style", "composition"):
        same = tot = 0
        for i in range(3):
            for j in range(i + 1, 3):
                for k in range(5):
                    same += arrs[i][k].get(f) == arrs[j][k].get(f); tot += 1
        out[a][f"k4_{f}"] = same / tot

names = {"zero": "zero-shot", "few": "few-shot", "step": "пошаговые инструкции"}
print(f"{'приём':<22}{'K1':>6}{'K2':>8}{'K3':>6}{'K4':>8}{'K4 style':>10}{'K4 comp':>9}")
for a, r in out.items():
    print(f"{names[a]:<22}{r['k1']:>6.2f}{r['k2']:>8.3f}{r['k3']:>6.2f}{r['k4']:>8.3f}{r['k4_style']:>10.3f}{r['k4_composition']:>9.3f}")
for a, r in out.items():
    for e in r["each"]:
        print(os.path.basename(e["file"]), f"K1={e['k1']:.1f} K2={e['k2']:.3f} K3={e['k3']:.1f}", "ошибки:", e["errors"] or "нет")
json.dump(out, open(os.path.join("reports", "k1k4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("записано: reports/k1k4.json")

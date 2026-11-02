# ЛР №9, вариант 2, часть 1. Яковенко Максим Михайлович.
# Скрипт Engee: ставит пакеты указаний, запускает Python-скрипт варианта (lr09_variant2.py) интерпретатором
# /user/.user_venvs/py3.11 и показывает вывод. Вычисления — в Python, Julia только управляет запуском.
# Рабочая папка между ячейками не сохраняется, поэтому все пути абсолютные.

# %% Шаг 3 плана. Среда: дата, версии, файлы работы и их SHA-256; пакеты указаний
using Dates, SHA
W = "/user/Яковенко_Максим_Михайлович/LR09"
PY = "/user/.user_venvs/py3.11/bin/python3"
println("Дата и время (UTC): ", now(UTC))
println("Julia VERSION = ", VERSION, ";  ядер: ", Sys.CPU_THREADS, ";  ОЗУ: ", round(Sys.total_memory() / 2^30; digits = 1), " ГБ")
for (f, ref) in (("lab09_lib.py", "cc4d233d"), ("lab09_llm.py", "5eabb31f"), ("run_lr09.py", "a962a37b"), ("lr09_variant2.py", ""))
    h = bytes2hex(open(sha256, joinpath(W, f)))
    println(rpad(f, 18), filesize(joinpath(W, f)), " байт, SHA-256 ", h, isempty(ref) ? "" : "; совпадает с архивом работы: $(startswith(h, ref))")
end
# Раздел 4 указаний: diffusers 0.32.2 и transformers < 5, PyTorch не обновлять. Если torch в окружении нет
# (так было в ЛР 8), ставится версия указаний — 2.4.1 для процессора. Вывод pip — в pip_install.log.
has(m) = success(ignorestatus(`$PY -c "import $m"`))
PIPLOG = joinpath(W, "pip_install.log")
pipi(args...) = run(pipeline(`$PY -m pip install --progress-bar off $args`; stdin = devnull, stdout = PIPLOG, stderr = PIPLOG, append = true))
has("torch") || (println("ставлю torch 2.4.1 (CPU)"); pipi("torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cpu"))
has("scipy") || (println("ставлю scipy"); pipi("scipy"))
ver = read(ignorestatus(`$PY -c "import diffusers, transformers; print(diffusers.__version__, transformers.__version__)"`), String)
startswith(ver, "0.32.2 4.") || (println("ставлю diffusers==0.32.2, transformers<5"); pipi("diffusers==0.32.2", "transformers<5"))
run(`$PY -c "import sys, torch, diffusers, transformers, scipy, numpy; print('python', sys.version.split()[0], '| torch', torch.__version__, '| diffusers', diffusers.__version__, '| transformers', transformers.__version__, '| scipy', scipy.__version__, '| numpy', numpy.__version__, '| cuda', torch.cuda.is_available())"`)

# %% Шаги 4–6 плана. Вариант 2: серия A и B на seed 101…105, сводка, проверки. Запуск блокирующий
# (фоновый запуск из ячейки подвешивает ячейку). Если results_v02.json уже есть, повторно не запускается.
cmd = Cmd(`$PY -u $(joinpath(W, "lr09_variant2.py"))`; dir = W)
if isfile(joinpath(W, "results_v02.json"))
    println("результат уже есть: results_v02.json от ", Dates.unix2datetime(mtime(joinpath(W, "results_v02.json"))), " UTC — повторно не запускаю")
else
    t0 = time(); run(pipeline(cmd; stdin = devnull, stdout = joinpath(W, "out_v02.txt"), stderr = joinpath(W, "err_v02.txt")))
    println("вариант 2 выполнен за ", round((time() - t0) / 60; digits = 1), " мин; stderr: ", filesize(joinpath(W, "err_v02.txt")), " байт")
end

# %% Вывод скрипта варианта целиком
print(read(joinpath(W, "out_v02.txt"), String))

# %% Шаг 5 плана. Лист сравнения seed 101: слева A, справа B
using Images
hcat(load(joinpath(W, "v02", "A_s101.png")), load(joinpath(W, "v02", "B_s101.png")))

# %% Шаг 7 плана. Запросы части 2 для новых чатов языковой модели (контекст 1)
for a in ("zero", "few", "step")
    println("=== ", a, " ===")
    run(Cmd(`$PY $(joinpath(W, "lab09_llm.py")) prompt $a 1`; dir = W))
end

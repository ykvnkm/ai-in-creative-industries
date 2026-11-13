# ЛР №11, вариант 2. Яковенко Максим Михайлович.
# Скрипт Engee: проверяет среду, запускает обучение под наблюдением (demo_gan.py указаний, вариант 2) и расчёт
# варианта (lr11_variant2.py) интерпретатором /user/.user_venvs/py3.11, показывает вывод и листы образцов.
# Вычисления — в Python, Julia только управляет запуском. Пути абсолютные: рабочая папка между ячейками не сохраняется.

# %% Шаг 3 плана. Среда: дата, версии, файлы работы и их SHA-256
using Dates, SHA
W = "/user/Яковенко_Максим_Михайлович/LR11"
PY = "/user/.user_venvs/py3.11/bin/python3"
println("Дата и время (UTC): ", now(UTC))
println("Julia VERSION = ", VERSION, ";  ядер: ", Sys.CPU_THREADS, ";  ОЗУ: ", round(Sys.total_memory() / 2^30; digits = 1), " ГБ")
for (f, ref) in (("lab07_lib.py", "d84b83a8"), ("lab11_lib.py", "88b6083f"), ("demo_gan.py", "f91864ac"), ("lr11_variant2.py", ""))
    h = bytes2hex(open(sha256, joinpath(W, f)))
    println(rpad(f, 18), filesize(joinpath(W, f)), " байт, SHA-256 ", h, isempty(ref) ? "" : "; совпадает с выданным: $(startswith(h, ref))")
end
# Раздел 4 указаний: нужен только PyTorch 2.4.1 (и NumPy); Pillow — для листов образцов demo_gan.py.
run(`$PY -c "import sys, torch, numpy, PIL; print('python', sys.version.split()[0], '| torch', torch.__version__, '| numpy', numpy.__version__, '| Pillow', PIL.__version__, '| cuda', torch.cuda.is_available())"`)

# %% Шаги 4–5 плана. Два независимых процесса по одному потоку (LAB_THREADS=1, как в прогоне преподавателя):
# demo/ — обучение режимов A и B на seed 101 со снимками на шагах 100, 300, 700, 1500 (demo_gan.py 11 2);
# основная папка — variant_summary(2), 10 обучений. Процессы запускаются в отдельной сессии (setsid):
# зависание вкладки или перезапуск ядра их не останавливают, результаты пишутся в файлы.
mkpath(joinpath(W, "demo"))
if isfile(joinpath(W, "results_v02.json"))
    println("результат уже есть: results_v02.json от ", Dates.unix2datetime(mtime(joinpath(W, "results_v02.json"))), " UTC — повторно не запускаю")
else
    sh = "cd $W/demo && LAB_THREADS=1 setsid nohup $PY -u ../demo_gan.py 11 2 > out_demo.txt 2> err_demo.txt < /dev/null & " *
         "cd $W && LAB_THREADS=1 setsid nohup $PY -u lr11_variant2.py > out_v02.txt 2> err_v02.txt < /dev/null &"
    run(`sh -c $sh`)
    println("запущено в фоне: ", now(UTC), " UTC")
end

# %% Ход расчёта (ячейку можно выполнять повторно)
for f in ("demo/out_demo.txt", "out_v02.txt", "err_v02.txt")
    p = joinpath(W, f)
    println("== ", f, isfile(p) ? "" : " — ещё нет")
    isfile(p) && print(join(last(readlines(p), 6), "\n"), "\n")
end

# %% Вывод скрипта варианта целиком
print(read(joinpath(W, "out_v02.txt"), String))

# %% Журнал обучения под наблюдением (demo_gan.py): метрики и потери на шагах 100, 300, 700, 1500
print(read(joinpath(W, "demo", "demo11_fmt.txt"), String))

# %% Шаг 6 плана. 32 образца seed 101: сверху режим A, снизу режим B
using Images
vcat(load(joinpath(W, "samples", "A_s101.png")), load(joinpath(W, "samples", "B_s101.png")))

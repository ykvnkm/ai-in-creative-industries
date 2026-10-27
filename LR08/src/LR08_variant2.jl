# ЛР №8, вариант 2. Яковенко Максим Михайлович.
# Скрипт Engee: запускает Python-скрипт варианта (lr08_variant2.py) и демонстрационный скрипт указаний (lab08_demo.py)
# интерпретатором /user/.user_venvs/py3.11 и показывает их вывод. Вычисления — в Python, Julia только управляет запуском.
# Рабочая папка между ячейками не сохраняется, поэтому все пути абсолютные.

# %% Шаг 1. Среда: дата, версии, файлы работы и их SHA-256
using Dates, SHA
W = "/user/Яковенко_Максим_Михайлович/LR08"
PY = "/user/.user_venvs/py3.11/bin/python3"
println("Дата и время (UTC): ", now(UTC))
println("Julia VERSION = ", VERSION, ";  ядер: ", Sys.CPU_THREADS, ";  ОЗУ: ", round(Sys.total_memory() / 2^30; digits = 1), " ГБ")
for (f, ref) in (("lab07_lib.py", "d84b83a8"), ("lab08_lib.py", "f414c49b"), ("lab08_demo.py", "8ab781bf"), ("lr08_variant2.py", ""))
    h = bytes2hex(open(sha256, joinpath(W, f)))
    println(rpad(f, 18), filesize(joinpath(W, f)), " байт, SHA-256 ", h, isempty(ref) ? "" : "; совпадает с указаниями: $(startswith(h, ref))")
end
# Окружение py3.11 обновлялось после проверки указаний, и torch в нём может не оказаться.
# Тогда ставится версия указаний — torch 2.4.1, сборка для процессора (графического ускорителя в Engee нет).
# Вывод pip пишется в pip_install.log: его можно читать, пока установка идёт.
has(m) = success(ignorestatus(`$PY -c "import $m"`))
PIPLOG = joinpath(W, "pip_install.log")
pipi(args...) = run(pipeline(`$PY -m pip install --progress-bar off $args`; stdin = devnull, stdout = PIPLOG, stderr = PIPLOG, append = true))
for (m, args) in (("numpy", ("numpy<2",)), ("PIL", ("Pillow",)), ("sympy", ("sympy",)),
                  ("torch", ("torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cpu")))
    has(m) && continue
    println("ставлю ", join(args, " ")); pipi(args...)
end
run(`$PY -c "import sys, torch, numpy, PIL; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| numpy', numpy.__version__, '| Pillow', PIL.__version__)"`)

# %% Шаг 3–8. Скрипт варианта 2 (3 потока). Запуск блокирующий: фоновый запуск из ячейки (wait = false) в Engee
# подвешивает ячейку. Если results_v02.json уже записан, скрипт повторно не запускается (обучение на процессоре
# не побитово воспроизводимо, и повтор заменил бы уже сохранённые результаты другими).
cmd_v2 = addenv(Cmd(`$PY -u $(joinpath(W, "lr08_variant2.py"))`; dir = W), "OMP_NUM_THREADS" => "3")
if isfile(joinpath(W, "results_v02.json"))
    println("результат варианта 2 уже есть: results_v02.json от ", Dates.unix2datetime(mtime(joinpath(W, "results_v02.json"))), " UTC — повторно не запускаю")
else
    t0 = time(); run(pipeline(cmd_v2; stdin = devnull, stdout = joinpath(W, "out_v02.txt"), stderr = joinpath(W, "err_v02.txt")))
    println("вариант 2 выполнен за ", round((time() - t0) / 60; digits = 1), " мин")
end

# %% Демонстрационный скрипт указаний как есть (вариант 1), в отдельной папке demo — его comparison.png не затрёт лист варианта
mkpath(joinpath(W, "demo"))
cmd_demo = addenv(Cmd(`$PY -u $(joinpath(W, "lab08_demo.py"))`; dir = joinpath(W, "demo")), "OMP_NUM_THREADS" => "4")
t0 = time(); run(pipeline(cmd_demo; stdin = devnull, stdout = joinpath(W, "demo", "out_demo.txt"), stderr = joinpath(W, "demo", "err_demo.txt")))
println("демонстрационный скрипт выполнен за ", round((time() - t0) / 60; digits = 1), " мин; stderr: ", filesize(joinpath(W, "demo", "err_demo.txt")), " байт")

# %% Вывод скрипта варианта 2 целиком (снимок вывода консоли для отчёта)
print(read(joinpath(W, "out_v02.txt"), String))

# %% Вывод демонстрационного скрипта (вариант 1, как есть)
print(read(joinpath(W, "demo", "out_demo.txt"), String))

# %% Шаг 6. Лист сравнения comparison.png варианта 2
using Images
load(joinpath(W, "comparison.png"))

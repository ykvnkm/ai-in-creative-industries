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
run(`$PY -c "import sys, torch, numpy, PIL; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| numpy', numpy.__version__, '| Pillow', PIL.__version__)"`)

# %% Шаг 3–8. Запуск двух процессов в фоне: вариант 2 (3 потока) и демонстрационный скрипт как есть (2 потока, отдельная папка demo)
mkpath(joinpath(W, "demo"))
cmd_v2 = addenv(Cmd(`$PY -u $(joinpath(W, "lr08_variant2.py"))`; dir = W), "OMP_NUM_THREADS" => "3")
cmd_demo = addenv(Cmd(`$PY -u $(joinpath(W, "lab08_demo.py"))`; dir = joinpath(W, "demo")), "OMP_NUM_THREADS" => "2")
P_V2 = run(pipeline(cmd_v2; stdin = devnull, stdout = joinpath(W, "out_v02.txt"), stderr = joinpath(W, "err_v02.txt")); wait = false)
P_DEMO = run(pipeline(cmd_demo; stdin = devnull, stdout = joinpath(W, "demo", "out_demo.txt"), stderr = joinpath(W, "demo", "err_demo.txt")); wait = false)
T0 = now(UTC)
println("запущено в ", T0, ": вариант 2 и демонстрационный вариант 1")

# %% Контроль хода: выполнять повторно, пока оба процесса не завершатся
for (name, p, out, err) in (("вариант 2", P_V2, joinpath(W, "out_v02.txt"), joinpath(W, "err_v02.txt")),
                            ("демо", P_DEMO, joinpath(W, "demo", "out_demo.txt"), joinpath(W, "demo", "err_demo.txt")))
    st = process_running(p) ? "работает" : "завершён, код $(p.exitcode)"
    lines = isfile(out) ? readlines(out) : String[]
    println("== ", name, ": ", st, "; прошло ", round((now(UTC) - T0).value / 60000; digits = 1), " мин; строк вывода ", length(lines))
    foreach(println, lines[max(1, end - 4):end])
    isfile(err) && filesize(err) > 0 && println("stderr: ", read(err, String)[max(1, end - 600):end])
end

# %% Вывод скрипта варианта 2 целиком (снимок вывода консоли для отчёта)
print(read(joinpath(W, "out_v02.txt"), String))

# %% Вывод демонстрационного скрипта (вариант 1, как есть)
print(read(joinpath(W, "demo", "out_demo.txt"), String))

# %% Шаг 6. Лист сравнения comparison.png варианта 2
using Images
load(joinpath(W, "comparison.png"))

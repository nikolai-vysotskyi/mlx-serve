# Historical local-session appendix — not the cloud bootstrap

Local paths are redacted placeholders. Start with README.md. Historical shell commands are for a configured Mac only.

# Qwen3.8-Flash-Next / M5 Max: handoff исследования prefill

Обновлено: **10 сентября 2026**. Этот файл — стартовая точка новой сессии. Это отчёт о состоянии и пересказ требований пользователя; фактические инструкции новой сессии и её AGENTS.md имеют приоритет. Старый HANDOFF из `/private/tmp` — исторический материал, а не актуальное задание.

## 1. Задача и критерий успеха

Пользователь хочет ускорить **prefill всей Qwen3.8-Flash-Next** на локальном **Apple M5 Max, 128 GB RAM** без ухудшения качества. Ближайшая цель: **больше 1,5× относительно актуального upstream**, затем 2×/3× и выше. Нужны измеренные токены/с и конкретный PR с изменениями, которые дают этот результат.

Последние уточнения пользователя:

- Исследовать крупные алгоритмические изменения; меньше повторных замеров небольших эффектов.
- Если отдельная находка даёт 1,1×, искать совместимые дополнительные изменения до общего ≥1,5×.
- **Не запускать тяжёлые full-model benchmark/test-серии ради небольшого компонента, пока нет обоснованного кандидата/набора на ≥1,5×.** Текущая работа над отдельным GDN PR приостановлена по этому уточнению.
- Фиксировать гипотезы, предложения, отрицательные и положительные результаты в существующих GitHub issues. Публикация, push, PR и уместные @пинги сопровождающих уже разрешены пользователем.
- Не объяснять отсутствие результата «пределом TFLOPS/железа». Искать, как уменьшить работу, повторные загрузки и преобразования, изменить алгоритм и организацию исполнения.
- Не выдавать ускорение одной операции за ускорение модели. Не складывать проценты разных стадий.

**Состояние цели: НЕ достигнута. Подтверждённого ≥1,5× на текущем main нет. Нового PR с таким результатом нет.**

Не публиковать заголовок «ускорили модель в 2,4×»: 2,41× ниже относится только к подготовке GDN, которая занимает небольшую часть общего времени.

## 2. С чего начать новой сессии

1. Прочитать этот README целиком, затем `outputs/prefill-investigation-2026-09-10.md`.
2. Прочитать применимые AGENTS.md, `CONTRIBUTING.md`, `CLAUDE.md`, релевантный `docs/gotchas/engine-mlx.md` выбранного checkout. Bench skill: `.claude/skills/bench/SKILL.md`.
3. Посмотреть свежие сообщения issue #366 и diff upstream с `cc7dea1`. Последний прочитанный HEAD — `fb15a8d8c3b489af31a898d38ee4f2ff8c4a26be` (`fix: opencode2 launcher`), но наши последние измерения относятся к `cc7dea1d18077ae3368570d1a0983c9a587613ac`.
4. Сверить `outputs/handoff-state-2026-09-10.json`: SHA checkout, ветки, незакоммиченные файлы, SHA бинарников. Не считать имя папки доказательством содержимого бинарника.
5. Выбрать набор гипотез с бюджетом сокращения **не менее трети полного времени**, лучше с запасом. Сначала исходники/формулы/короткие диагностические фикстуры; не начинать с ещё одного длинного benchmark.
6. Продолжать работу в этой папке. Не создавать новую пользовательскую задачу автоматически: пользователь собирался открыть новую сессию сам.

В `<historical-user-AGENTS.md>` требуется trace-mcp first для навигации по коду. `get_project_map` работает; Zig-файлы этих checkout часто не индексируются. `search`/`get_outline` возвращали not found/not indexed, тогда использовался `rg` и точечные чтения. Не тратить время на одинаковые безрезультатные trace-запросы. Подагенты в текущей сессии не разрешены без явного запроса; новая сессия должна соблюдать собственные инструкции.

## 3. Пути, модель и железо

Все относительные пути далее — от:

```sh
TASK_ROOT=<historical-mac-workspace>
cd "$TASK_ROOT"
```

Другие пути:

```sh
SCRATCH_ROOT=<historical-mac-scratchpad>
MODEL_ROOT=<historical-model-directory>/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit
```

- Model/API ID: `ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit`.
- macOS 26.5, Apple M5 Max, 128 GB. Модель — `qwen4_exp`, а не произвольная Qwen3.5.
- Серверный **MLX 0.32.3**, локальная pinned сборка с NAX. Python/pip MLX может отличаться; его нельзя незаметно подменить в измерениях.
- Zig: `0.17.0-dev.1818+7051f8e73`, `work/mlx-serve-next/.zig-toolchain/zig`.
- Веса ~73,4 GB десятичных; MLX active после загрузки ~71,39 GB. Отдельная PLE mmap-таблица ~29,8 GB; это не ещё одна полностью resident MLX allocation.
- 48 слоёв: 36 GDN, 12 full attention. Hidden=2560.
- MoE: 512 экспертов, top-10, intermediate=640, основные эксперты affine4/group64/bf16 scales/biases.
- GDN: 16 key heads, 48 value heads, Dk=Dv=128, conv kernel=4, conv_dim=10240.
- HC: 4 потока, lowrank=320, ширина потока 4×2560=10240.
- QSA: 24 query heads, 2 KV heads, D=256; indexer 4×128, compress ratio=4, budget=2048.
- Квантизация разрешается per tensor. Не переносить предположение affine4 на все веса без проверки shapes/scales/biases/config.

## 4. Репозитории: что где находится

| Путь в `work/` | Содержание и назначение |
|---|---|
| `mlx-serve` | Старые первоначальные эксперименты. Не основной baseline. Содержит исходник, из которого harness берёт фиксированный корпус. |
| `mlx-serve-local` | Старый combined build: QSA + exact MoE reduction + parallel PLE + conv compaction и другие локальные изменения. Исторический выигрыш vs v26.9.2. Есть незакоммиченные/неиспользуемые эксперименты. Не отправлять весь diff в PR. |
| `mlx-serve-qsa-pr` | Наш прежний QSA PR #385; CLOSED, код включён через #388. Не открывать дубликат. |
| `mlx-serve-upstream-2692` | Чистый `1ec580a` / v26.9.2. Исторический baseline, не current main. |
| `mlx-serve-conv-pr` | Изолированная conv-state compaction на v26.9.2; code+tests+HTTP проверялись. PR ещё не создан. Экономия памяти, отдельного speedup нет. |
| `mlx-serve-next` | Checkout `cc7dea1`, но сейчас **с незакоммиченной conv compaction**. Его текущий `zig-out/bin` не является чистым main! |
| `main-cc7dea1` | Сохранённый **чистый бинарник cc7dea1**, файл, не каталог. Подходит как исторический current-main control. |
| `mlx-serve-gateup-next` | `cc7dea1` + эксперимент fused MoE gate/up/SwiGLU, opt-in `MLX_SERVE_MOE_PREFILL_GATEUP=1`. ReleaseFast собран, HTTP/full-model проверялся. Нет подтверждённого общего выигрыша; полного suite для этой ветки нет. |
| `mlx-serve-gdn-prefill` | `cc7dea1` + незавершённый opt-in wide GDN prework/norm-gate. **Есть numeric test failure; готового бинарника нет. Не использовать как подтверждённое ускорение.** |

Ключевые SHA-256:

- `work/main-cc7dea1`: `3fc2c7ccb44b925086cebfedc62ce81a8e20136d6fd3e22f377c59c6422bfa49`.
- Gate/up binary: `f2437f436d7503e729092afd9b7d61c58fcfb48fd40646325b15feb8190604a6`.
- Текущий `mlx-serve-next/zig-out/bin/mlx-serve` (conv patch): `8cce6cf3ae5278cc09fb157cac08031e3a1dbc2bbb33bb824fa427d0f62abbd5`.

**Критическая ловушка:** прежде общий symlink `.zig-cache` между checkout возвращал устаревшие бинарники. Старые `qsa-main-*` отчёты с этой атрибуцией недостоверны. У каждого checkout должен быть СОБСТВЕННЫЙ реальный `.zig-cache`. Ссылки на pinned toolchain/библиотеки допустимы. Совпадение текста `--version` недостаточно.

Некоторые submodule dirs в новых локальных клонах — пустые. Проверять необходимые файлы, не только существование каталога. `lib/opencode2-mlx-serve/LICENSE` нужен для сборки cc7. В GDN checkout пустой каталог уже заменён ссылкой на имеющийся локальный submodule; **после этого сборка не повторялась**. Общий `git status` может ругаться на .git внутри symlink submodule: `git status --short --ignore-submodules=all` показывает наш diff.

## 5. Подтверждённые и неподтверждённые цифры

### Последние полные прогоны — без заявленного выигрыша

Одинаковое железо, chunk=8192, cache entries=0, MTP off. llmprobe 0.6.6, `--rungs 64k --runs 1`:

| Отчёт `outputs/<tag>-llmprobe.json` | Реальных input tokens | Long prefill tok/s | TTFT ms |
|---|---:|---:|---:|
| `main-cc7dea1-retry` | 68651 | 1545 | 44445 |
| `gateup-cc7-on` | 68742 | 1691 | 40656 |
| `gateup-cc7-off` | 68651 | 1706 | 40249 |

Вывод: брать 1545 как единственный знаменатель и объявлять +9% неверно; последующий control оказался быстрее кандидата. Сам llmprobe калибрует длину немного по-разному — реальные counts надо сообщать.

На отдельном **фиксированном source-корпусе 65836 tokens**: gate/up on **1611,2**, off **1513,4 tok/s**, по одному наблюдению, пароль восстановлен, cached=0. +6,5% — только предварительное наблюдение, противоречащее общему выводу llmprobe. Не тратить ещё 10 минут на повторение этих процентов: пользователь изменил приоритет.

`bench.prefillTokPerSec` в JSON — **короткий тест ~2041 tokens**. Нужная длинная метрика находится в **`bench.contextScaling[].prefillTokPerSec`**, рядом `inputTokens` и `ttftMs`. 1850/1935 tok/s короткого теста нельзя присвоить gate/up: ядро требует ≥4096 tokens в чанке и там вообще не включается.

`gateup-cc7-on-repeat` был намеренно прерван по просьбе пользователя. Не использовать частичный прогон как завершённый результат.

### Исторический успех — уже частично вошёл в upstream

Старый combined build: 1679–1693 tok/s против более быстрого control v26.9.2 1460, **+15–16%**. Это не ≥1,5× и не сравнение с cc7. Наш QSA код уже принят через #388, поэтому нельзя прибавлять его выигрыш повторно.

Исходные отчёты: `outputs/{upstream-2692-8k,compact-8k-vs-2692}{,-repeat}-llmprobe.json`.

### Где сейчас уходит время

Один инструментированный проход `work/main-cc7-investigation-profile-8192-49000.log`, JSON `outputs/main-cc7-block-profile.json`. Профайлер синхронизирует GPU на границах блоков: это диагностика, не baseline скорости.

Чанк S=8192 при KV=40960:

| Стадия | мс | Примерная доля суммы |
|---|---:|---:|
| MLP/MoE | 1621,6 | 34,7% |
| GDN | 1186,4 | 25,4% |
| Attention | 1084,9 | 23,2% |
| HC read+write | 693,9 | 14,8% |
| PLE block | 89,4 | 1,9% |

Отдельный PLE gather логируется отдельно. Первый/второй чанк имели холодные эффекты PLE, поэтому не брать их как steady разбивку. Сумма этих блоков не включает весь HTTP/host overhead.

Бюджет для выбора исследований (НЕ прогноз): если удастся MLP 1,5×, GDN 2×, attention 1,5× и HC 1,5×, указанная сумма уменьшится примерно в **1,59×**. Запас небольшой: реальный полный результат должен быть проверен. Формула: `speedup = sum(t_i) / sum(t_i / s_i)`. Если две оптимизации касаются одного времени, их выигрыши нельзя учесть дважды.

## 6. Эксперименты: не повторять отвергнутое вслепую

| Идея | Что установлено | Статус |
|---|---|---|
| MoE fused gate/up/SwiGLU, BM64 WM4 WN2 | ~22–26% на отдельной цепочке, exact в фикстурах; full-model gain не подтверждён | Сохранить как небольшой компонент, не центр исследования |
| Direct input gather в NAX | Удаляет replicated input 419430400 B при S8192, но 22,08 против stock 17,37 ms | Отложен |
| То же с threadgroup staging | 23,76 против 17,44 ms | Отложен |
| Однократная dequantize gate/up + dense gather_mm | 37,91 против 17,21 ms включая распаковку, exact | Отложен |
| MoE inverse gather + weight + top10 reduce | 2,31–2,47 → 0,674 ms на S4096, 10,49M outputs exact | Небольшой совместимый компонент в старом local tree |
| Conv-state compaction | 83,59 → 77,85 GB peak; isolated prefill 1433 → 1410 tok/s | Память, не скорость; не включать в speed budget без нового эффекта |
| Chunk32768 после compaction | Уже помещается, но 1491/1527 tok/s — медленнее | Не повторять «просто увеличить chunk» |
| GDN fewer lanes/row | 4,316 → 4,119 ms на S4096, небольшой numeric drift | Не крупный рычаг |
| Gate/up BM64 WM2 WN2 | ~52 ms от register pressure | Отвергнут |
| Gate/up WM2 WN4 | Ошибка max diff 28,625 | НЕ использовать |
| BK128 при quant group64 | Нарушает ограничение quant loader | Не включать без переработки loader |
| PLE pooled pread при resident table | Ранее проигрывал ~2–7% | Не считать log SERIAL доказательством отсутствия старого parallel mmap |

Компонентные .cpp/.metal и результаты лежат в `work/`: `grouped_gateup*`, `direct_gateup*`, `staged_gateup*`, `dequant_gateup*`, `gdn_bench.cpp`, `gdn-results.jsonl`.

### Незавершённый GDN prefill — ВАЖНО

Файлы:

- `work/gdn_prefill_prework_bench.cpp`, собранный executable с тем же stem.
- `work/gdn_prework_source.metal`, `gdn_kernel_header.metal`, `gdn_normgate_source.metal`: извлечены из upstream Zig, не новый алгоритм.
- `work/gdn-prefill-prework-8k.jsonl`: **3,91867 → 1,62408 ms (2,41×)**, шесть выходов exact на random [-1,1], full model geometry, S8192.
- Это экономит ориентировочно 2,3 ms × 36 слоёв ≈83 ms на чанк, то есть само по себе далеко от целевого сокращения ~1,5 секунды.

В `work/mlx-serve-gdn-prefill` изменены gates dispatch:

- `MLX_SERVE_GDN_PREFILL_FUSED=1` разрешает подходящим Qwen4 prefill вызовам prework/norm-gate на 10..8192 rows; default off.
- decode ветка сохранена; projection GEMM и recurrent kernel не менялись.
- prework требует `ssm.initialized`; первый чанк может идти composed. Norm-gate разрешён независимо от этого.
- Расширены существующие Zig numerical tests на S17/128/4096 (prework), S17/128/8192 (norm-gate), включая folded layouts и batch.
- Новый HTTP script `tests/test_gdn_prefill_fusions.py`, ещё НЕ запускался.

**Проверки не зелёные:**

1. Red до dispatch-изменения: test `gdn packed prework` отказался с `FusedDeclined` на широкой последовательности — ожидаемо (`work/gdn-prefill-red.log`).
2. После изменения полный suite: **2307 passed, 153 skipped, 1 failed**, `work/gdn-prefill-tests.log`. Ошибка: exact comparison `pre.beta` vs MLX sigmoid, **max diff 0,0000076293945**. Тест использует b/a с более широким диапазоном (~[-8,8]), чем первый C++ benchmark. Не скрывать это tolerances, не утверждать exact/no-quality-loss, не открывать готовый PR. Подозрение на bf16 sigmoid/exp/округления — причина НЕ установлена. Возможное направление: LUT sigmoid, уже применённый в exact SwiGLU; сначала установить конкретные несовпадающие inputs.
3. ReleaseFast не собрался: отсутствовал `lib/opencode2-mlx-serve/LICENSE`. Пустой submodule dir уже исправлен ссылкой на имеющуюся локальную копию. **Повторной сборки не было**, готового GDN server binary нет.
4. Full-model измерений этой ветки не было. Тяжёлые проверки НЕ возобновлять ради изолированного 2% общего выигрыша; пользователь уточнил приоритет ≥1,5× набора.

### HC up-projection + mixing: написан, НЕ запускался

`work/hc_up_mix.metal`, `work/hc_up_mix_bench.cpp`, собран `work/hc_up_mix_bench`.

Прототип вычисляет четыре HC up-проекции с NAX и сразу sigmoid × normalized stream + mean, без сохранения полного up `[8192,4,2560]` (~168 MB). Reference — **реальный compiled HC mix callback эквивалент**, не просто удобная несфузированная цепочка.

GPU correctness/speed НЕ проверены. Код специализирован на M8192, HC4, H2560, R320, BM32; нет общего handling хвостов. Точность mean/reduction и sigmoid под compile нужно проверить. Не интегрировать вслепую. Даже большой локальный выигрыш должен войти в общий бюджет, а не стать причиной новой длинной серии процентов.

## 7. Как искать именно кратные рычаги

Каждая гипотеза должна содержать:

1. **Какую существующую работу устраняем:** количество операций, проходов по памяти, повторных деквантизаций, синхронизаций или последовательных шагов.
2. Где она вызывается на **prefill**, при фактических shapes/quant/dtype этой модели. Decode-only оптимизация не считается.
3. Сколько времени этой области показывает профиль и какой общий выигрыш получается при реалистичном сокращении. Не подменять это числом TFLOPS.
4. Почему новый алгоритм сохранит значения: формула, dtype промежуточных результатов, порядок округлений, state carry между чанками.
5. С чем сочетается; где выигрыши перекрываются; какая память/компиляция/копии добавляются.
6. Дешёвый способ опровергнуть гипотезу до загрузки 74-GB модели.

Направления для следующего существенного исследования:

- **GDN chunkwise/WY вместо последовательного state update по всему T.** Нынешний blocked-seq kernel лишь staging блоков, внутри всё ещё идёт цикл по токенам. Матричная chunkwise форма может перенести работу на NAX, но recurrence — только часть GDN; нужно оценивать вместе с projections/prework. Смена порядка float-сумм требует численной проверки, не автоматически bit-exact.
- **Совместная оптимизация projections/layout/epilogues в GDN и HC.** Простое поднятие HC_FUSED_MAX_ROWS до8192 неверно: его D/U — decode GEMV. Сохранить NAX GEMM. Новое HC prototype — только один участок этой цепи.
- **MoE grouped GEMM и reuse внутри эксперта:** нужны существенные изменения tile reuse/scheduling, а не повтор unsorted-vs-sorted (сортировка УЖЕ есть). Не тащить случайные row reads внутрь NAX без новой причины — два варианта уже проиграли.
- **QSA attention и общая организация chunking.** Наш gather NAX и новый fused score upstream уже входят в baseline. Искать повторное использование выбранных блоков/данных, совместимость с query/kv группами и узкие места selection; не уменьшать budget2048, context или top-k ради цифр.
- Память сама по себе не throughput. Conv compaction полезна только если открывает иной эффективный алгоритм/расписание.

Первичные источники для алгоритмического разбора (CUDA speedup не переносится на M5 как готовый результат):

- Qwen FlashQLA: https://github.com/QwenLM/FlashQLA — algebraic reformulation, fusion, context parallelism для GDN. Реализация CUDA/TileLang; читать идеи/формулы, не обещать прямой запуск.
- FLA GDN chunk: https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
- Исходная работа Gated Delta Networks: https://arxiv.org/abs/2412.06464
- Delta rule sequence parallelism: https://arxiv.org/abs/2406.06484

Читать источники точечно. Если они используют затухание для приближённого усечения, нельзя молча перенести это как точное вычисление. Для сохранения качества не удалять экспертов, веса, состояния, внимание или токены.

## 8. Воспроизведение: только после обоснования крупного кандидата

### GPU и процессы

`work/bench_prefill.py` сам занимает GPU lock, поднимает СВОЙ сервер 11234, проверяет готовность, затем останавливает его и снимает lock. Если порт занят чужим сервером — отказывается продолжать. Не делать `pkill mlx-serve`, не убивать чужие задачи. Пользователь ранее освобождал память, но текущее состояние проверять заново. Не обходить admission без фактов.

Для отдельного GPU executable:

```sh
"$SCRATCH_ROOT/gpu_lock.sh" status
"$SCRATCH_ROOT/gpu_lock.sh" acquire codex-my-experiment || exit 1
trap '"$SCRATCH_ROOT/gpu_lock.sh" release codex-my-experiment' EXIT
# Здесь ОДИН собственный эксперимент; этот shell остаётся живым.
```

Замок хранит PID родительского shell; если acquire и GPU run идут из разных shell, защита не работает. Не удалять живой lock. Stale lock снимается скриптом сам.

В этой сессии GPU/MLX, `ps`, network и сборки обычно требовали tool escalation. Это техническое разрешение среды; авторизация пользователя на исследование и публикацию уже дана. Запускать необходимые разрешённые действия, не переспрашивать каждые пять минут.

### Компиляция компонентных прототипов

Из `$TASK_ROOT`, пример (не команда на немедленный запуск):

```sh
clang++ -std=c++20 -mmacosx-version-min=26.2 -O3 work/hc_up_mix_bench.cpp \
  -Iwork/mlx-serve-next/lib/mlx/include \
  -Lwork/mlx-serve-next/lib/mlx/lib -lmlx \
  -Wl,-rpath,"$TASK_ROOT/work/mlx-serve-next/lib/mlx/lib" \
  -o work/hc_up_mix_bench
```

Проверять actual library path. Во многих .cpp есть `#include "qsa_bench.cpp"` с переименованием main — это существующие random/read helpers, не запуск QSA. Metal-файлы читаются относительно `$TASK_ROOT`. Большинство фикстур синтетические; использовать несколько релевантных диапазонов/маршрутизаций, но не устраивать большую сетку повторов ради процентов.

### Полный prefill: одинаковые условия A/B

Только когда есть обоснованный набор на ≥1,5×:

```sh
python3 work/bench_prefill.py \
  --binary "$TASK_ROOT/work/main-cc7dea1" \
  --chunks 8192 --tokens 49000 --reps 1 \
  --nax on --reduce off --compact off --gateup off --gdn off \
  --llmprobe --probe-runs 1 --tag baseline-UNIQUE
```

Для candidate заменить binary, включить ТОЛЬКО реализованные флаги, дать другое уникальное имя tag. **Не использовать GDN checkout как candidate сейчас: у него нет готового бинарника и failing numeric test.** Если есть более новый upstream с performance changes, собрать свежий clean baseline с собственной cache, вместо cc7 backup.

- `--tokens 49000` — приблизительная цель по символам harness; фактически fixed-source prompt =65836 tokens. В записи всегда читать `usage`, не аргумент CLI.
- `--llmprobe` запускает pinned 0.6.6, длинный rung64k; чтение `bench.contextScaling` описано выше.
- `--reps` — дополнительные fixed-corpus HTTP запросы ПОСЛЕ llmprobe. Не включать reps3/probe-runs3 по умолчанию: пользователь просит экономить время.
- Один A/B обнаруживает крупный эффект; если ≥1,5× есть, сделать короткое подтверждение/обратный порядок и обязательные correctness checks перед заявлением. Не публиковать одиночный шум как доказательство.
- Cache entries=0, реальные cached tokens=0, одинаковые chunk/ctx/spec/power/thermal conditions. Профилирующие sync должны быть выключены в speed comparison.
- Harness выставляет `QWEN4_PLE_PAR=16`, но на cc7 без локального parallel-PLE кода это не реализованный feature. Env сам по себе ничего не доказывает.
- Harness по умолчанию **reduce=on, nax=off, probe-runs=3**. Воспроизводить с явными флагами, как выше, а не полагаться на defaults.
- Наличие в env не доказывает engagement. Проверить собственные server logs: QSA NAX, MoE gate/up `[moe-prefill-gateup] engaged: rows=81920`, GDN `[gdn] packed prework ... S=...`. На off отсутствие маркера обязательно, на on нужен размер из соответствующего prefill пути.
- `--profile` включает QWEN4_PROFILE_FWD=all; применять лишь для диагностики. Один профиль уже сохранён, повторять без новой причины не нужно.

### Проверки/PR после сильного кандидата

Читать актуальный CONTRIBUTING. Репозиторий требует compile + полный `zig build test` на реальном Mac перед PR, red/green regression, HTTP script и реальные клиентские запросы. Сейчас пользователь просит не выполнять этот тяжёлый этап ради небольших компонентов. Поэтому до крупного кандидата публиковать исследования в issue; PR не объявлять готовым и не обходить numeric failures.

Пример будущих команд из выбранного checkout:

```sh
DYLD_LIBRARY_PATH="$SCRATCH_ROOT/fresh/lib/mlx/lib:$SCRATCH_ROOT/fresh/lib/llama/lib" \
  .zig-toolchain/zig build test
.zig-toolchain/zig build -Doptimize=ReleaseFast
```

Сначала проверять exit code каждого шага; не продолжать автоматически после failed tests. Сохранить полный лог, commit SHA, patch, binary SHA-256, library version. Полный suite использует GPU — тоже под lock. Для доказывания exactness применять широкие реальные диапазоны, нестандартные хвосты, batch, state reuse, dtype/quant fallback. Правильный пароль на длинном prompt — полезный HTTP smoke, но не общая оценка качества.

## 9. GitHub: где продолжать

Upstream: https://github.com/ddalcu/mlx-serve

- Основное открытое исследование: https://github.com/ddalcu/mlx-serve/issues/366
- Research checkpoint с бюджетом ≥1,5× и отвергнутыми гипотезами: https://github.com/ddalcu/mlx-serve/issues/366#issuecomment-5617958072
- Последний handoff update с GDN beta failure и незапущенным HC prototype: https://github.com/ddalcu/mlx-serve/issues/366#issuecomment-5620422178
- Старое измеренное +15–16% vs v26.9.2: https://github.com/ddalcu/mlx-serve/issues/366#issuecomment-5599721751
- QSA accepted via https://github.com/ddalcu/mlx-serve/pull/388 ; наш https://github.com/ddalcu/mlx-serve/pull/385 закрыт.
- PLE #375 уже существует. System-message issue #365 исправлен upstream. Не открывать дубликаты.

CLI `gh` авторизован как `nikolai-vysotskyi`. Fork: `nikolai-vysotskyi/mlx-serve`. Сопровождающие: `ddalcu`, `beamivalice`. Перед новым issue/PR искать open+closed. Использовать `gh search ... --repo ddalcu/mlx-serve`, не строку с неверно закавыченным `repo:...`.

Для публикации body писать в локальный UTF-8 файл, затем `gh issue comment ... --body-file ...` / `gh pr create --body-file ...`. Никаких shell-подстановок через JSON.stringify. Не делать force push. Локальный origin некоторых checkout указывает на другую папку, а не GitHub; проверять remote перед push.

Последнее требование пользователя: **конкретный PR, который ускоряет общий prefill >1,5×**. Не заменять его PR с одной операцией, ускоренной в 2,4×. Пока результата нет — честные промежуточные notes в issue, как уже разрешено.

## 10. Состояние на момент передачи

- Подтверждённого нового ≥1,5× нет.
- Тяжёлых benchmark/build/test процессов этой работы при последней проверке нет; не предполагается продолжение в фоне.
- GDN suite завершился с одним numeric failure; build завершился ошибкой зависимости, локальная ссылка исправлена без пересборки.
- HC prototype только скомпилирован как C++ executable, GPU не запускался.
- Новых PR из этой сессии нет. Research checkpoint в #366 опубликован.
- `outputs/start-flash-next.sh` указывает на старый local build; не считать, что пользовательский запуск переключён на текущие эксперименты.
- `outputs/RESULTS.md` и `outputs/prefill-progress-2026-09-09.json` местами устарели; свежие точные данные здесь и в handoff JSON.
- Все экспериментальные исходники и логи сохранены. Ничего не requantized/repacked в постоянных весах модели.

## 11. Сообщение для новой сессии

Скопировать следующий текст, указав этот файл:

> Продолжай исследование локальной Qwen3.8-Flash-Next на M5 Max 128 GB. Прочитай `<historical-mac-workspace>/README.md` целиком и затем актуальные инструкции репозитория. Цель: измеренное ускорение prefill ВСЕЙ модели больше 1,5× относительно актуального upstream без ухудшения качества и PR с этим набором изменений. Подтверждённого результата пока нет. Не трать время на долгие прогоны 1–10% компонентов: ищи крупные алгоритмические рычаги и совместимый набор, сначала обоснуй бюджет сокращения общего времени. Веди исследование в issue #366, публикация и PR разрешены. Учитывай failing GDN beta parity, уже принятый upstream QSA и ловушку shared Zig cache. Не повторяй отвергнутые эксперименты без новой причины. Работай в существующей папке и сохраняй воспроизводимые артефакты.

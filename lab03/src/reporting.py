"""Regenerate the graded figures and Ukrainian report from actual saved runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"MLP": "#2864b7", "CNN": "#18834b", "Attention": "#b55716"}
SUITE_LABELS = {"reference": "Exact manual architectures", "matched": "Models with similar parameter counts"}


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        for key, value in record.items():
            try:
                record[key] = float(value)
            except ValueError:
                pass
    return records


def save_figure(figure, results, name):
    figure.tight_layout()
    figure.savefig(results / f"{name}.png", dpi=180, bbox_inches="tight")
    svg_path = results / f"{name}.svg"
    figure.savefig(svg_path, bbox_inches="tight")
    # Matplotlib leaves spaces at the ends of SVG path lines. Normalize them so
    # regenerated, versioned figures also pass git diff --check.
    svg_path.write_text("\n".join(line.rstrip() for line in
                                   svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    plt.close(figure)


def make_plots(results, groups):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    count = len(groups)
    figure, axes = plt.subplots(1, count, figsize=(6.5 * count, 4.7), squeeze=False)
    for axis, (suite, records) in zip(axes[0], groups.items()):
        for row in records:
            x, y = row["parameters"] / 1000, 100 * row["test_accuracy"]
            axis.scatter(x, y, s=95, color=COLORS[row["model"]], zorder=3)
            axis.annotate(f"{row['model']}\n{int(row['parameters']):,}; {y:.2f}%", (x, y),
                          xytext=(6, 9), textcoords="offset points", fontsize=9)
        if suite == "reference":
            axis.axvspan(100, 150, color="#7790a8", alpha=0.12, label="100k-150k target")
            axis.set_xlim(0, max(row["parameters"] for row in records) / 1000 * 1.3)
            axis.legend(loc="lower right")
        else:
            axis.set_xlim(117, 122)
        accuracy = [100 * row["test_accuracy"] for row in records]
        axis.set_ylim(min(accuracy) - 2, max(accuracy) + 4)
        axis.set(title=SUITE_LABELS[suite], xlabel="Trainable parameters (thousands)", ylabel="Test accuracy (%)")
        axis.grid(alpha=0.22)
    save_figure(figure, results, "accuracy_vs_parameters")

    figure, axes = plt.subplots(count, 2, figsize=(12.8, 4.6 * count), squeeze=False)
    for pair, (suite, records) in zip(axes, groups.items()):
        for axis, batch in zip(pair, (1, 128)):
            for row in records:
                x, y = row[f"latency_b{batch}_ms"], 100 * row["test_accuracy"]
                axis.errorbar(x, y, xerr=[[x - row[f"latency_b{batch}_p25_ms"]],
                                          [row[f"latency_b{batch}_p75_ms"] - x]],
                              fmt="o", markersize=8, capsize=4, color=COLORS[row["model"]])
                axis.annotate(f"{row['model']} ({x:.3f} ms)", (x, y), xytext=(5, 10),
                              textcoords="offset points", fontsize=9)
            max_latency = max(row[f"latency_b{batch}_p75_ms"] for row in records)
            accuracy = [100 * row["test_accuracy"] for row in records]
            axis.set(xlim=(0, max_latency * 1.45), ylim=(min(accuracy) - 2, max(accuracy) + 4),
                     title=f"{SUITE_LABELS[suite]} | batch {batch}",
                     xlabel=f"Median forward latency per batch (ms), B={batch}; bars: IQR",
                     ylabel="Test accuracy (%)")
            axis.grid(alpha=0.22)
    save_figure(figure, results, "accuracy_vs_latency")

    figure, axes = plt.subplots(count, 2, figsize=(12.8, 4.3 * count), squeeze=False)
    for pair, (suite, records) in zip(axes, groups.items()):
        for row in records:
            history = read_csv(results / suite / f"{row['model'].lower()}_history.csv")
            epochs = [record["epoch"] for record in history]
            pair[0].plot(epochs, [record["train_loss"] for record in history], "-o",
                         label=row["model"], color=COLORS[row["model"]])
            pair[1].plot(epochs, [100 * record["validation_accuracy"] for record in history], "-o",
                         label=row["model"], color=COLORS[row["model"]])
        for axis in pair:
            axis.set_xticks(range(1, 6))
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.22)
            axis.legend()
        pair[0].set(title=SUITE_LABELS[suite], ylabel="Online training cross-entropy")
        pair[1].set(title=SUITE_LABELS[suite], ylabel="Validation accuracy (%)")
    save_figure(figure, results, "training_curves")

    figure, axes = plt.subplots(1, count, figsize=(6.5 * count, 4.7), squeeze=False)
    for axis, (suite, records) in zip(axes[0], groups.items()):
        positions = np.arange(len(records))
        for shift, key, label, color in (
            (-0.24, "hand_flops", "Hand: core FLOPs", "#2864b7"),
            (0, "thop_raw_ops", "2 x raw THOP output (incomplete)", "#b55716"),
            (0.24, "thop_core_flops", "THOP: aligned core FLOPs", "#18834b"),
        ):
            axis.bar(positions + shift, [row[key] * (2 if key == "thop_raw_ops" else 1) / 1e6
                                         for row in records], width=0.23, label=label, color=color)
        axis.set_xticks(positions, [row["model"] for row in records])
        axis.set(title=SUITE_LABELS[suite], ylabel="Million operations per image, B=1")
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.22)
    save_figure(figure, results, "flops_comparison")


def metrics_table(records):
    lines = [
        "| Модель | Параметри | Core FLOPs | Ваги FP32, KiB | Test accuracy | B=1, ms | B=128, ms/пакет | B=128, ms/зображення |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(f"| {row['model']} | {int(row['parameters']):,} | {int(row['hand_flops']):,} "
                     f"| {row['weights_kib']:.2f} | {100 * row['test_accuracy']:.2f}% "
                     f"| {row['latency_b1_ms']:.5f} | {row['latency_b128_ms']:.5f} "
                     f"| {row['latency_b128_per_image_ms']:.5f} |")
    return "\n".join(lines)


def make_report(results, groups, config):
    sections = [f"""# Lab 3: Architectures and their costs

## Мета та відтворення

Порівняти MLP, CNN та attention за точністю Fashion-MNIST, кількістю параметрів,
арифметичною вартістю, затримкою inference та характером доступу до пам'яті.
Усі числа нижче отримані виконанням коду; точність не є оцінкою чи прикладом.

З кореня репозиторію:

```powershell
python main.py
```

За замовчуванням виконуються **обидва** експерименти: точні схеми з інструкції
(`reference`) та окремі моделі однакового порядку розміру (`matched`).
Кожна з шести моделей навчається рівно 5 епох. Окремі варіанти запуску:

```powershell
python main.py --suite reference
python main.py --suite matched
python main.py --report-only
.\\.venv\\Scripts\\python.exe lab03/src/check_lab.py
```

Launcher використовує поточний Python або готову локальну `.venv`, якщо у поточному
немає PyTorch. Дані завантажуються через `torchvision.datasets.FashionMNIST` у
`data/`, що виключено з Git. За недоступності історичного HTTP-сервера код
використовує HTTPS-дзеркало авторів з перевіркою оригінальних MD5.

## Дані й протокол

- Офіційні 60 000 навчальних зображень розділено на **55 000 train / 5 000 validation**.
  Офіційні **10 000 test** залишені окремо. Індекси збережені в `results/split_indices.npz`.
- Перетворення точно за умовою: `ToTensor()` і `Normalize((0.2860,), (0.3530,))`.
  Зображення мають форму `[B,1,28,28]`; мітки — цілі числа 0–9.
- Batch size **128**, `drop_last=False`, без augmentation. Для всіх моделей однакові
  індекси train/validation і послідовності перемішування пакетів.
- **Adam, lr=0.001, 5 епох**, стандартні betas `(0.9,0.999)`, eps `1e-8`,
  weight decay 0; `CrossEntropyLoss` отримує logits без попереднього softmax.
- Seed **{config['seed']}** для Python, NumPy, PyTorch, поділу й перемішування.
  Детерміновані алгоритми увімкнено. Validation обчислюється після кожної епохи;
  test accuracy фіксується для фінальної 5-ї епохи, без вибору за test.
- Середовище: **{config['device_name']}**, `{config['device']}`, FP32,
  intra-op threads **{config['torch_threads']}**, inter-op threads **{config['torch_interop_threads']}**.
  Python {config['python'].split()[0]}, PyTorch {config['packages']['torch']},
  torchvision {config['packages']['torchvision']}, torchinfo {config['packages']['torchinfo']},
  THOP {config['packages']['thop']}, NumPy {config['packages']['numpy']},
  Matplotlib {config['packages']['matplotlib']}. Повна конфігурація — `results/configuration.json`.

## Архітектури та суперечність в інструкції

Точні розміри шарів з п. 2 **несумісні** з одночасною вимогою 100–150 тисяч
параметрів. Це перевірено ручним підрахунком, `torchinfo.summary` та
`sum(p.numel() for p in model.parameters())`. Щоб покрити обидві вимоги,
точні архітектури збережено окремо від експерименту з близькою кількістю параметрів.

| Експеримент | MLP | CNN | Attention | Кількість параметрів |
|---|---|---|---|---|
| reference | 784→256→128→10, ReLU | Conv 1→16→32, 3×3, padding 1, pool 2 після кожної; Linear 1568→10 | 16 патчів 7×7, d=64, heads=4, FF=128, 2 encoders | 235 146 / 20 490 / 70 794 |
| matched | 784→128→128→10, ReLU | Conv 1→48→128; ті самі kernel, padding та pooling; Linear 6272→10 | d=84, heads=4, FF=168, 2 encoders; ті самі 16 патчів | 118 282 / 118 634 / 119 794 |

У `matched` змінено тільки ширини; глибина та родини операцій збережені.
Максимальна різниця кількості параметрів між моделями — близько 1.28% від мінімальної.
Це дає контрольованіше порівняння, але однакова кількість параметрів не означає
однакові FLOPs чи однакову точність.

Attention: `[B,1,28,28] → [B,16,49] → Linear(49,d) → 2 TransformerEncoderLayer
→ mean(dim=1) → Linear(d,10)`. Encoder має ReLU, dropout 0.1, залишкові зв'язки
та два LayerNorm, `batch_first=True`, post-norm; кожний encoder ініціалізується
окремо. Mean pooling не змішує різні зображення пакета.
У схемі з інструкції немає positional embedding або class token, тому вони
не додані: після mean pooling модель інваріантна до перестановки 16 патчів.
Позиції пікселів усередині кожного патча збережені, взаємне розташування патчів — ні.
Це обмеження враховано при поясненні точності.

## Ручний підрахунок параметрів

Bias включено всюди. Для Linear: `in*out + out`; для Conv:
`K*K*Cin*Cout + Cout`; LayerNorm(d) має `2*d` trainable коефіцієнтів.

- MLP: `(784*256+256) + (256*128+128) + (128*10+10)`
  `= 200960 + 32896 + 1290 = 235146`.
- CNN: `(3*3*1*16+16) + (3*3*16*32+32) + (1568*10+10)`
  `= 160 + 4640 + 15690 = 20490`.
- Attention projection: `49*64+64 = 3200`; один encoder:
  QKV `3*64*64+3*64 = 12480`, output `64*64+64 = 4160`,
  FF1 `64*128+128 = 8320`, FF2 `128*64+64 = 8256`,
  два LayerNorm `4*64 = 256`. Разом `33472` на encoder;
  уся модель `3200 + 2*33472 + (64*10+10) = 70794`.

Усі покомпонентні підрахунки, включно з `matched`, збережені в
`results/<suite>/hand_calculations.csv`; форми тензорів і таблиці параметрів —
`results/<suite>/*_summary.txt` з `torchinfo.summary`.

## Ручні FLOPs та порівняння з THOP

Узгоджена конвенція: **одне множення + одне додавання = 2 FLOPs**,
batch=1. `Core FLOPs` тут рахують матричні добутки та згортки.
Bias, ReLU, порівняння MaxPool, softmax, LayerNorm, residual additions та
mean pooling не включені. Це представницька арифметична вартість,
а не виміряні інструкції процесора або повна вартість усіх операцій.
Формули згортки рахують також операції на нульовому padding.

Репрезентативні шари точних архітектур:

```text
MLP Linear(784,256):
  2 * 784 * 256 = 401408 FLOPs.
  Вся MLP: 401408 + 65536 + 2560 = 469504.

CNN Conv2d(1,16,3,padding=1), вихід 16x28x28:
  2 * 3 * 3 * 1 * 16 * 28 * 28 = 225792 FLOPs.
  Conv2 (32x14x14): 2 * 3 * 3 * 16 * 32 * 14 * 14 = 1806336.
  Linear: 2 * 1568 * 10 = 31360. Разом: 2063488.

Attention, один encoder: L=16, d=64, h=4, head_dim=16, FF=128:
  QKV:             2 * 16 * 64 * (3 * 64) = 393216
  QK^T:            2 * h * L^2 * (d/h)   = 32768
  Attention @ V:   2 * h * L^2 * (d/h)   = 32768
  Output proj:     2 * 16 * 64 * 64      = 131072
  FF1 + FF2:       2 * (2 * 16 * 64 * 128) = 524288
  Один encoder: 1114112 FLOPs.
  Projection 16 * Linear(49,64): 100352; classifier: 1280.
  Вся Attention: 100352 + 2 * 1114112 + 1280 = 2329856.
```

Кількість голів не додає ще одного множника `h` поверх `d`: сума розмірностей
голів дорівнює `d`. Для двох attention encoder повна формула core FLOPs:
`2*L*49*d + 2*(8*L*d^2 + 4*L^2*d + 4*L*d*FF) + 2*d*10`.

`thop.profile` повертає MACs для Linear/Conv, тому їх множимо на два.
У зафіксованій версії немає повного правила для `nn.MultiheadAttention`;
крім того, LayerNorm рахується в інших одиницях. Вивід «2 × raw THOP» для
attention є лише діагностикою і не може вважатися повними FLOPs.
Збережено початковий результат і окреме профілювання з власним hook для MHA,
який за фактичними розмірами входу рахує QKV, scores, weighted values і output
projection. LayerNorm виключено для узгодження з ручними core FLOPs.

Під час профілювання вимикається fused MHA/encoder fastpath: у PyTorch 2.3
він обходить module hooks. Потім попередній стан відновлюється; звичайний
inference benchmark використовує оптимізації PyTorch. THOP запускається на
копіях моделей, тому його службові buffers не потрапляють у checkpoint.
"""]
    for suite, records in groups.items():
        sections.append(f"\n### {suite}: виміряні результати\n\n" + metrics_table(records))
        lines = ["\n| Модель | Ручні core FLOPs | THOP raw ops | 2×raw (діагностика) | THOP core FLOPs | Різниця core | THOP raw params / torchinfo |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for row in records:
            lines.append(f"| {row['model']} | {int(row['hand_flops']):,} | {int(row['thop_raw_ops']):,} "
                         f"| {int(2 * row['thop_raw_ops']):,} | {int(row['thop_core_flops']):,} "
                         f"| {row['core_gap_percent']:.2f}% | {int(row['thop_raw_parameters']):,} / {int(row['torchinfo_parameters']):,} |")
        sections.append("\n".join(lines))
        layer_rows = read_csv(results / suite / "representative_layers.csv")
        layer_table = ["\nНезалежне профілювання окремого шару з фінального checkpoint:",
                       "\n| Шар | Ручні FLOPs | THOP MACs | THOP FLOPs |",
                       "|---|---:|---:|---:|"]
        for layer in layer_rows:
            layer_table.append(f"| {layer['layer']} | {int(layer['hand_flops']):,} "
                               f"| {int(layer['thop_macs']):,} | {int(layer['thop_flops']):,} |")
        sections.append("\n".join(layer_table))
        attention = next(row for row in records if row["model"] == "Attention")
        d = 64 if suite == "reference" else 84
        missing_mha_macs = 2 * (4 * 16 * d * d + 2 * 16 * 16 * d)
        norm_raw_ops = 4 * 4 * 16 * d
        sections.append(f"\nДля {suite} Attention точний розклад розбіжності: "
                        f"raw THOP {int(attention['thop_raw_ops']):,} = "
                        f"core MACs {int(attention['thop_core_macs']):,} − "
                        f"невраховані MHA MACs {missing_mha_macs:,} + "
                        f"LayerNorm ops {norm_raw_ops:,}. Raw parameter count також пропускає "
                        "ваги MHA. Власне правило відновлює повний підрахунок параметрів і core MACs.")
    sections.append("""
## Протокол inference latency

Для кожної вже навченої моделі: `eval()` і `torch.inference_mode()`, 20 warmup
проходів окремо для кожного batch size. Далі **медіана 100 запусків для B=1**
та **медіана 20 запусків для B=128**. Таймер — `perf_counter_ns()`.
Вхід — однакові перші 128 test-зображень; потрібний пакет уже підготовлений
і перенесений на пристрій до таймера. Таймер охоплює повний forward моделі,
включно з patchify та mean pooling; завантаження даних, transforms і transfer
виключені. Для CUDA виконується synchronization перед та після кожного запуску.
Вимірювання проводяться послідовно при однакових налаштуваннях потоків.

Кожен запуск є окремим рядком у `results/<suite>/latency_runs.csv`.
`metrics.csv` містить median, 25-й і 75-й перцентилі, ms/зображення та throughput.
На графіках горизонтальні відрізки означають IQR latency; це не довірчий
інтервал accuracy. B=128 показано як час **усього пакета**, а ms/зображення
обчислюється діленням на 128.

## Графіки

![Accuracy against parameters](results/accuracy_vs_parameters.png)

![Accuracy against inference latency, batches 1 and 128](results/accuracy_vs_latency.png)

![Hand and profiled FLOPs](results/flops_comparison.png)

![Training and validation curves](results/training_curves.png)

Також збережено SVG кожного графіка для масштабування без втрати якості.

## Архітектурні компроміси та доступ до пам'яті (§4.6–4.7)

**MLP.** Dense Linear виконує регулярні matrix-vector/matrix-matrix добутки.
При B=1 арифметична інтенсивність мала: кожна вага використовується для одного
зображення. При B=128 ті самі ваги використовуються для багатьох прикладів,
що поліпшує reuse і завантаження обчислювальних блоків. Для першого reference
Linear, ігноруючи cache та інший трафік, нижня оцінка читання ваг —
`784*256*4 = 802816` байтів на 401408 FLOPs при B=1, тобто близько
0.5 FLOP/byte; при повному reuse на B=128 — близько 64 FLOP/byte щодо самих ваг.
Це модель арифметичної інтенсивності, а не апаратне вимірювання DRAM-трафіку.
MLP ігнорує просторову структуру зображення і витрачає багато параметрів на
перший шар, зате має коротку послідовність операцій.

**CNN.** Sliding windows використовують просторову локальність; один фільтр
повторно застосовується в кожній позиції. Мала кількість ваг може відповідати
великій кількості FLOPs. Для reference CNN 20 490 параметрів забезпечують
2 063 488 core FLOPs, бо ваги Conv багаторазово повторно використовуються.
Pooling зменшує просторовий розмір наступних активацій. Трафік включає карти
ознак: перший reference Conv створює `16*28*28=12544` float32 елементів,
тобто 49 KiB на зображення, а не лише 80.04 KiB ваг усієї мережі.
У matched CNN більші channel dimensions збільшують FLOPs і проміжні тензори
значно сильніше, ніж у MLP того самого розміру. Реальні бібліотеки можуть
використовувати tiling чи тимчасові buffers; обсяг останніх тут не вимірювався.

**Attention.** QKV і FFN — регулярні dense GEMM, attention додає глобальні
залежності між патчами, softmax та редукції. Для L=16, h=4 логічна матриця
scores має `h*L*L=1024` елементи, 4 KiB на зображення на encoder
(512 KiB для B=128), а QKV — `3*L*d` елементів. Із ростом L score-вартість
зростає квадратично. Для короткої послідовності цього експерименту проєкції
та FFN переважають у core FLOPs. Перестановки, читання/запис активацій,
LayerNorm, residual connections і виклики багатьох операторів створюють
затримку, яку параметри чи core FLOPs самі по собі не описують.
Scores — логічний обсяг, а не виміряний peak RAM: fused kernels можуть уникати
повної матеріалізації деяких тензорів. Щільний attention реалізовано регулярними
матричними операціями; content-dependent weights не означають випадкові
апаратні gathers. Без positional embedding модель втрачає порядок патчів,
що разом із коротким навчанням є можливим поясненням її точності.

У термінах підручника: MLP використовує регулярні доступи й reuse ваг у пакеті;
CNN додає sliding-window locality і weight sharing; attention комбінує dense
проєкції, broadcast/reduction та глобальне змішування інформації. У всіх трьох
архітектурах data movement, cache locality, operator fusion і паралелізм
впливають на latency. За самою кількістю FLOPs не можна встановити, чи ця
конкретна мережа memory-bound або compute-bound: для цього потрібні hardware
counters або roofline-профілювання. Патерни вище є аналізом структури алгоритмів.
""")
    sections.append("\n## Висновки за фактичними вимірюваннями\n")
    for suite, records in groups.items():
        best = max(records, key=lambda row: row["test_accuracy"])
        fastest = min(records, key=lambda row: row["latency_b1_ms"])
        fastest_batch = min(records, key=lambda row: row["latency_b128_per_image_ms"])
        smallest = min(records, key=lambda row: row["parameters"])
        cnn = next(row for row in records if row["model"] == "CNN")
        mlp = next(row for row in records if row["model"] == "MLP")
        sections.append(
            f"**{suite}.** Найвища test accuracy у {best['model']} "
            f"({100 * best['test_accuracy']:.2f}%). Найменша B=1 latency у {fastest['model']} "
            f"({fastest['latency_b1_ms']:.5f} ms); найменше параметрів у {smallest['model']} "
            f"({int(smallest['parameters']):,}). CNN проти MLP: різниця точності "
            f"{100 * (cnn['test_accuracy'] - mlp['test_accuracy']):+.2f} відсоткових пунктів, "
            f"відношення core FLOPs {cnn['hand_flops'] / mlp['hand_flops']:.2f}×, "
            f"відношення B=1 latency {cnn['latency_b1_ms'] / mlp['latency_b1_ms']:.2f}×."
        )
        pareto = [row["model"] for row in records if not any(
            other["test_accuracy"] >= row["test_accuracy"] and
            other["parameters"] <= row["parameters"] and
            other["latency_b1_ms"] <= row["latency_b1_ms"] and (
                other["test_accuracy"] > row["test_accuracy"] or
                other["parameters"] < row["parameters"] or
                other["latency_b1_ms"] < row["latency_b1_ms"])
            for other in records)]
        sections.append(f"За трьома критеріями (більша accuracy, менші parameters і B=1 latency) "
                        f"на Pareto-межі {suite}: **{', '.join(pareto)}**. "
                        "Вибір залежить від обмеження на час відповіді та потрібної точності.")
        sections.append(
            f"Практичний вибір у цьому експерименті: **{best['model']} для максимальної "
            f"точності**, якщо допустимі {best['latency_b1_ms']:.5f} ms на одиночний forward; "
            f"**{fastest['model']} для мінімальної B=1 затримки**. Для пакетної обробки "
            f"найменший час на зображення також окремо перевірено: {fastest_batch['model']}, "
            f"{fastest_batch['latency_b128_per_image_ms']:.5f} ms при B=128."
        )
        dominated = [row["model"] for row in records if row["model"] not in pareto]
        if dominated:
            sections.append(f"Моделі **{', '.join(dominated)}** мають альтернативу з не гіршою "
                            "accuracy, не більшою кількістю параметрів та не більшою B=1 latency "
                            "(і хоча б один критерій кращий), тому за цими критеріями їхній вибір "
                            "у цьому запуску не виправданий.")
        for row in records:
            speedup = row["latency_b1_ms"] / row["latency_b128_per_image_ms"]
            sections.append(f"- {suite} {row['model']}: B=128 дає {speedup:.2f}× "
                            f"менший середній час на зображення, ніж B=1; "
                            f"throughput {row['throughput_b128_images_s']:.0f} зображень/с.")
    sections.append("""

Для цих невеликих зображень CNN має корисний просторовий inductive bias та
weight sharing; її доцільність оцінюється разом з отриманою accuracy і latency.
MLP доцільна за жорсткого обмеження на затримку, якщо її точності достатньо.
Глобальний attention має додаткову системну вартість, а наведена схема без
позиційної інформації і лише 5 епох не гарантує переваги на Fashion-MNIST.
Matched-експеримент показує, чи зберігаються ці висновки при близькому числі ваг.

Обмеження: один seed, 5 епох, один пристрій і одна реалізація inference;
accuracy не усереднена за кількома навчаннями. Фонове навантаження, power/thermal
режим і стан cache можуть змінити latency. FP32 weights KiB не є загальним
споживанням RAM. Висновки стосуються цього протоколу, а не всіх CNN/MLP/Transformer.

## Збережені докази та перевірки

- `results/configuration.json`, `results/split_indices.npz`: середовище, seed, повний протокол та поділ.
- `results/<suite>/metrics.csv`: параметри, ручні й THOP FLOPs, accuracy, latency, throughput.
- `results/<suite>/hand_calculations.csv`: формули й результати для кожного компонента.
- `results/<suite>/representative_layers.csv`: незалежні THOP-профілі одного Linear, Conv та MHA.
- `results/<suite>/*_summary.txt`: `torchinfo.summary`, форми входів/виходів, параметри.
- `results/<suite>/*_history.csv`: усі 5 епох, train loss/accuracy, validation loss/accuracy.
- `results/<suite>/latency_runs.csv`: 100+20 сирих вимірювань для кожної моделі.
- `results/<suite>/*.pt`: фінальні ваги з model spec, seed, epoch і test accuracy.
- Чотири графіки у PNG та SVG: accuracy–parameters, accuracy–latency, FLOPs, learning curves.

`check_lab.py` перевіряє форми та градієнти всіх моделей, неперекриття патчів,
підрахунок параметрів, узгодженість core FLOPs, масштабування профілю batch size,
інваріантність attention до порядку патчів і відновлення inference fastpath.
Сам runner додатково зупиняється при розбіжності ручних/профільованих підрахунків
або нечисловому loss. Завантажені дані та середовище в commit не включаються.
Перед створенням графіків автоматично перевіряються поділ, повнота епох,
відновлення checkpoint, кількість latency-запусків та їхні медіани.
Результат перевірки — `results/verification.json`.

## Джерела

1. Vijay Janapa Reddi, *Introduction to Machine Learning Systems*, Chapter 4,
   §4.2, §4.3, §4.5, §4.6–4.7 у редакції, зазначеній в інструкції.
   [Авторський онлайн-підручник: DNN architectures](https://mlsysbook.ai/vol1/nn_architectures/nn_architectures.html).
   Нумерація в актуальній онлайн-редакції може відрізнятися.
2. [torchvision 0.18: FashionMNIST](https://docs.pytorch.org/vision/0.18/generated/torchvision.datasets.FashionMNIST.html).
3. [THOP: MACs, parameters та custom hooks](https://github.com/Lyken17/pytorch-OpCounter).
4. [PyTorch: TransformerEncoderLayer та inference fastpath](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).
5. [torchinfo: model summary](https://github.com/TylerYep/torchinfo).
""")
    (results.parent / "report.md").write_text("\n\n".join(sections).strip() + "\n", encoding="utf-8")


def generate_outputs(results: Path):
    config = json.loads((results / "configuration.json").read_text(encoding="utf-8"))
    groups = {suite: read_csv(results / suite / "metrics.csv") for suite in config["suites"]}
    for suite, records in groups.items():
        if {row["model"] for row in records} != {"MLP", "CNN", "Attention"} or len(records) != 3:
            raise ValueError(f"Incomplete results for {suite}; finish training before generating the report.")
        for row in records:
            history = read_csv(results / suite / f"{row['model'].lower()}_history.csv")
            if [record["epoch"] for record in history] != list(range(1, 6)):
                raise ValueError(f"Expected five completed epochs for {suite}/{row['model']}.")
    from validation import verify_evidence
    verify_evidence(results, groups, config)
    make_plots(results, groups)
    make_report(results, groups, config)

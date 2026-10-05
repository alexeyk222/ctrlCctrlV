# Вернём CTRL+C / CTRL+V при работе с VDI на macOS

При работе через VDI общий буфер обмена между локальным Mac и удалённым рабочим столом может быть отключён.

Этот проект добавляет два сценария:

```text
VDI → Mac
выделить область экрана → OCR → локальный clipboard

Mac → VDI
локальный clipboard → эмуляция клавиатурного ввода → VDI
```

Основные горячие клавиши:

| Комбинация                | Действие                                                                  |
| ------------------------- | ------------------------------------------------------------------------- |
| `Ctrl + Option + Cmd + 4` | Выделить область → быстрый OCR (Qwen3.5-4B, Apple GPU) → clipboard Mac      |
| `Ctrl + Option + Cmd + 5` | Выделить область → самый точный OCR с отступами кода (chandra-ocr-2) → clipboard |
| `Ctrl + Option + Cmd + V` | Напечатать содержимое clipboard в активном окне через эмуляцию клавиатуры |
| `Ctrl + Option + Cmd + R` | Перезагрузить конфигурацию Hammerspoon                                    |

---

# 1. Что используется

Для работы нужны:

* **Hammerspoon** — горячие клавиши, screenshot, clipboard и эмуляция клавиатуры;
* **Apple Command Line Tools / Swift** — сборка OCR;
* встроенный в macOS **Vision Framework** — OCR без дополнительных зависимостей;
* **PaddleOCR PP-OCRv5** — выбранный OCR; Apple Vision доступен как запасной режим.

Выбор движка и ограничения описаны в [разборе OCR](docs/OCR_OPTIONS.md).

---

# 2. Клонирование репозитория

```bash
git clone https://github.com/alexeyk222/ctrlCctrlV.git ~/ctrlCctrlV
cd ~/ctrlCctrlV
```

Структура проекта:

```text
ctrlCctrlV/
├── init.lua
├── scripts/
│   ├── ocr.swift
│   ├── ocr_paddle.py
│   └── ocr_worker.py
├── docs/
│   └── OCR_OPTIONS.md
└── README.md
```

Назначение файлов:

* `init.lua` — конфигурация Hammerspoon;
* `scripts/ocr.swift` — OCR на Apple Vision;
* `scripts/ocr_paddle.py` — OCR на PaddleOCR.
* `scripts/ocr_worker.py` — фоновый процесс PaddleOCR, который сохраняет модели в памяти между снимками.

---

# 3. Установка Hammerspoon

Если установлен Homebrew:

```bash
brew install --cask hammerspoon
```

После установки запустить:

```text
Applications → Hammerspoon
```

В верхней панели macOS появится значок Hammerspoon.

---

# 4. Разрешения macOS

Hammerspoon необходимо дать системные разрешения.

## Accessibility

Открыть:

```text
System Settings
→ Privacy & Security
→ Accessibility
```

Включить:

```text
Hammerspoon
```

Это необходимо для эмуляции клавиатурного ввода.

## Screen Recording

Открыть:

```text
System Settings
→ Privacy & Security
→ Screen & System Audio Recording
```

Включить:

```text
Hammerspoon
```

Это необходимо для захвата области экрана.

После изменения разрешений желательно полностью перезапустить Hammerspoon.

---

# 5. Установка Command Line Tools

Проверить наличие:

```bash
xcode-select --print-path
```

Обычно должно быть:

```text
/Library/Developer/CommandLineTools
```

Проверить Swift:

```bash
xcrun swiftc --version
```

Если Command Line Tools не установлены:

```bash
xcode-select --install
```

---

# 6. Проверка Swift

Перед сборкой OCR рекомендуется проверить, что Swift и macOS SDK совместимы.

```bash
cat >/tmp/test.swift <<'EOF'
import Foundation
print("Foundation OK")
EOF

xcrun swiftc /tmp/test.swift -o /tmp/test && /tmp/test
```

Ожидаемый результат:

```text
Foundation OK
```

Если появляется ошибка:

```text
this SDK is not supported by the compiler
Please select a toolchain which matches the SDK
```

значит версии Swift compiler и macOS SDK несовместимы.

На тестовом Mac с:

```text
macOS 15.1
```

проблема возникала с:

```text
Command Line Tools 16.2
```

и была устранена установкой:

```text
Command Line Tools for Xcode 16.1
```

Скачать нужную версию можно на Apple Developer Downloads:

```text
https://developer.apple.com/download/all/?q=Command%20Line%20Tools%2016.1
```

После установки повторите тест `Foundation OK`.

---

# 7. Установка файлов проекта

Создать рабочий каталог Hammerspoon:

```bash
mkdir -p ~/.hammerspoon/scripts
```

Скопировать конфигурацию:

```bash
cp ~/ctrlCctrlV/init.lua \
   ~/.hammerspoon/init.lua
```

Скопировать OCR source:

```bash
cp ~/ctrlCctrlV/scripts/ocr.swift \
   ~/.hammerspoon/scripts/ocr.swift

cp ~/ctrlCctrlV/scripts/ocr_paddle.py \
   ~/.hammerspoon/scripts/ocr_paddle.py

cp ~/ctrlCctrlV/scripts/ocr_worker.py \
   ~/.hammerspoon/scripts/ocr_worker.py

cp ~/ctrlCctrlV/scripts/ocr_coreml.py ~/ctrlCctrlV/scripts/prepare_coreml.py \
   ~/.hammerspoon/scripts/
```

---

# 8. Сборка OCR

Создать отдельный Swift module cache:

```bash
rm -rf ~/.cache/swift-ocr
mkdir -p ~/.cache/swift-ocr
```

Собрать OCR:

```bash
xcrun swiftc \
  -module-cache-path ~/.cache/swift-ocr \
  -framework Vision \
  ~/.hammerspoon/scripts/ocr.swift \
  -o ~/.hammerspoon/scripts/ocr
```

Проверить, что файл появился:

```bash
ls -lh ~/.hammerspoon/scripts/ocr
```

Проверить запуск:

```bash
~/.hammerspoon/scripts/ocr
```

Ожидаемый результат:

```text
Usage: ocr <image>
```

---

# 9. Загрузка конфигурации Hammerspoon

Через меню Hammerspoon:

```text
Hammerspoon
→ Reload Config
```

При успешной загрузке появится:

```text
VDI Tools loaded ✓
```

В дальнейшем конфигурацию можно перезагружать комбинацией:

```text
Ctrl + Option + Cmd + R
```

## Установка PaddleOCR

Профили клавиш задаются в `OCR_KEY_PROFILES` в `init.lua`: `fast`, `accurate`, `paddle`
(PaddleOCR-VL-1.6) или `classic` (PP-OCRv5 / Apple Vision). Веса VL-моделей скачиваются заранее:

```bash
~/.hammerspoon/.venv-vl/bin/hf download mlx-community/Qwen3.5-4B-MLX-4bit
~/.hammerspoon/.venv-vl/bin/hf download mlx-community/chandra-ocr-2-oQ8
```

Сравнение моделей на реальном скриншоте — в [docs/VL_RESULTS.md](docs/VL_RESULTS.md).

На Apple Silicon используйте Python 3.9–3.13. Для данного Mac подходит `python3.12`. Создайте окружение и установите CPU-версию по [инструкции PaddlePaddle для macOS](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html):

```bash
python3.12 -m venv ~/.hammerspoon/.venv
~/.hammerspoon/.venv/bin/python -m pip install paddlepaddle==3.3.0
~/.hammerspoon/.venv/bin/python -m pip install paddleocr==3.7.0
~/.hammerspoon/.venv/bin/python -m pip install onnxruntime==1.30.0 onnx==1.23.1
~/.hammerspoon/.venv/bin/python -c 'import paddle; import paddleocr; paddle.utils.run_check()'
```

В текущем `init.lua` выбран `OCR_ENGINE = "paddle"`, а `PADDLE_ACCELERATOR = "auto"` на Apple Silicon направляет тяжёлый детектор текста в Core ML с `CPUAndNeuralEngine`. Распознавание символов остаётся в ONNX Runtime CPU: на тесте перенос этой части в Core ML менял знаки исходного кода. Если Core ML не скомпилирует модель, скрипт автоматически запускает обе части через быстрый ONNX CPU.

`PADDLE_LANGUAGE = "ru"` выбирает модель для русского и английского; для исключительно английского кода поставьте `"en"`. Hammerspoon заранее запускает один фоновый процесс и держит модели в памяти. Максимальная сторона детектора по умолчанию — 4000 пикселей; настройка `OCR_MAX_SIDE` (минимум 960). Для возврата к PaddlePaddle CPU задайте `PADDLE_ACCELERATOR = "paddle"`, для ONNX CPU — `"onnx"`, для Apple Vision — `OCR_ENGINE = "vision"`.

Детектор автоматически преобразуется в FP16 MLProgram, исходная модель сохраняется. Модели и компиляции хранятся в `~/.cache/ctrlcctrlv-coreml`; в памяти остаются четыре размера кадра, на диске — примерно восемь. Первый новый размер требует компиляции и может занимать несколько секунд. Повторный OCR больших тестовых кадров занимает около 0,8 с. Профилирование подтвердило назначение операций на Neural Engine. Результаты и ограничения точности: [испытания ANE](docs/ANE_RESULTS.md).

Проверить фоновый процесс:

```bash
ps -axo pid,ppid,etime,command | rg '[o]cr_worker.py'
```

В консоли Hammerspoon после прогрева выводится выбранный backend, например:

```text
PaddleOCR worker ready: Core ML FP16 detector (CPU + ANE) / ONNX CPU recognizer
```

Проверка на одном и том же изображении:

```bash
~/.hammerspoon/scripts/ocr image.png
~/.hammerspoon/.venv/bin/python ~/.hammerspoon/scripts/ocr_paddle.py image.png --lang ru
```

---

# 10. Использование

## VDI → Mac

Нажать:

```text
Ctrl + Option + Cmd + 4
```

Далее:

1. выделить область экрана с текстом или кодом;
2. дождаться сообщения:

```text
OCR → Clipboard ✓
```

3. вставить результат на Mac обычным:

```text
Cmd + V
```

Схема:

```text
VDI
 ↓
Screenshot
 ↓
Apple Vision или PaddleOCR
 ↓
Mac clipboard
```

---

## Mac → VDI

Скопировать текст на Mac обычным способом:

```text
Cmd + C
```

Поставить курсор в нужное место внутри VDI и нажать:

```text
Ctrl + Option + Cmd + V
```

Hammerspoon не использует VDI clipboard — он последовательно генерирует события клавиатуры и «печатает» текст в активное окно.

`Return` и `Tab` отправляются как отдельные нажатия после отпускания клавиш горячей комбинации. Для VDI с потерей символов увеличьте `TYPING_DELAY` и `KEY_PRESS_DELAY_US` в `init.lua`. Если целевой редактор не добавляет отступ после Enter, установите `REMOVE_EDITOR_AUTO_INDENT = false`; по умолчанию скрипт выделяет автоматически добавленный отступ через `Shift+Home`, затем вводит отступ из буфера. В VS Code проверьте результат на небольшом фрагменте с несколькими строками до переноса большого файла.


---

# 11. Обновление проекта

После изменения `init.lua`:

```bash
cp ~/ctrlCctrlV/init.lua \
   ~/.hammerspoon/init.lua
```

После этого:

```text
Ctrl + Option + Cmd + R
```

После изменения `scripts/ocr.swift`:

```bash
cp ~/ctrlCctrlV/scripts/ocr.swift \
   ~/.hammerspoon/scripts/ocr.swift
```

и пересобрать OCR:

```bash
xcrun swiftc \
  -module-cache-path ~/.cache/swift-ocr \
  -framework Vision \
  ~/.hammerspoon/scripts/ocr.swift \
  -o ~/.hammerspoon/scripts/ocr
```

Reload Hammerspoon после изменения только `ocr.swift` не требуется.

После изменения OCR скопируйте `ocr_paddle.py`, `ocr_worker.py`, `ocr_coreml.py` и `prepare_coreml.py` в `~/.hammerspoon/scripts/` и перезагрузите конфигурацию Hammerspoon; повторная сборка Swift не нужна.

---

# 12. Быстрая диагностика

## Hotkey не работает

Проверить:

```text
System Settings
→ Privacy & Security
→ Accessibility
→ Hammerspoon = ON
```

---

## Не появляется выбор области

Проверить:

```text
System Settings
→ Privacy & Security
→ Screen & System Audio Recording
→ Hammerspoon = ON
```

---

## `OCR engine not installed`

Проверить:

```bash
ls -lh ~/.hammerspoon/scripts/ocr
```

Если выбран Vision и файла нет — повторить сборку из раздела **«Сборка OCR»**. Если выбран PaddleOCR — проверьте `~/.hammerspoon/.venv/bin/python`, `~/.hammerspoon/scripts/ocr_paddle.py` и `~/.hammerspoon/scripts/ocr_worker.py`.

---

## Ошибка Swift / SDK

Проверить:

```bash
xcrun swiftc --version
xcrun --show-sdk-path
```

И выполнить тест:

```bash
xcrun swiftc /tmp/test.swift -o /tmp/test && /tmp/test
```

Если появляется:

```text
this SDK is not supported by the compiler
```

необходимо установить совместимую версию Command Line Tools.

---

# 13. Ограничения

OCR хорошо подходит для переноса текста и большинства фрагментов кода, однако распознавание исходного кода не гарантирует 100% точность.

Автоматические подмены кавычек и округление отступов до четырёх пробелов удалены: оба движка оставляют распознанные символы без таких исправлений. Восстановление пробелов по координатам OCR остаётся приблизительным, особенно при пропорциональном шрифте и нескольких колонках.

Особенно внимательно стоит проверять:

```text
,
.
:
;
()
[]
{}
_
|
```

Мелкая пунктуация и отдельные строки вроде:

```text
)
```

иногда могут быть пропущены OCR.

Перед выполнением перенесённого кода рекомендуется быстро проверить результат.

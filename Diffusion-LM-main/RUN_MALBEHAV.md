# Diffusion-LM Runbook For MalBehavD-V1

Ниже полный набор запусков под ваш датасет `MalBehavD-V1` с комментариями:
- для чего команда;
- чем отличаются разные запуски одного и того же скрипта.

Все команды выполняются из папки `Diffusion-LM-main`.

## 0) Установка окружения

```powershell
pip install -e .
pip install spacy==3.2.4
pip install datasets==1.8.0
pip install huggingface_hub==0.4.0
pip install wandb
```

Назначение:
- установка локального пакета и зависимостей для train/sample/nll.

## 1) Подготовка MalBehav в ROC-формат (`prepare_malbehav_roc.py`)

### 1.1 Смешанный набор (рекомендуемый основной кейс)

```powershell
python scripts/prepare_malbehav_roc.py `
  --input_csv ../MalbehavD-V1-main/MalBehavD-V1-dataset.csv `
  --out_dir diffusion_lm/ROCstory `
  --valid_size 500 `
  --label_filter all `
  --seed 101
```

Для чего:
- делает train/valid из всех классов (`0` и `1`) для одной общей генеративной модели.

### 1.2 Только malware-класс

```powershell
python scripts/prepare_malbehav_roc.py `
  --input_csv ../MalbehavD-V1-main/MalBehavD-V1-dataset.csv `
  --out_dir diffusion_lm/ROCstory_malware `
  --valid_size 300 `
  --label_filter malware `
  --seed 101
```

Для чего:
- отдельная генерация только вредоносных API-последовательностей.

Отличие от 1.1:
- `--label_filter malware` вместо `all`;
- обычно меньше `valid_size`.

### 1.3 Только benign-класс

```powershell
python scripts/prepare_malbehav_roc.py `
  --input_csv ../MalbehavD-V1-main/MalBehavD-V1-dataset.csv `
  --out_dir diffusion_lm/ROCstory_benign `
  --valid_size 300 `
  --label_filter benign `
  --seed 101
```

Для чего:
- отдельная генерация только benign API-последовательностей.

Отличие от 1.2:
- меняется только фильтр класса и выходная папка.

## 2) Обучение Diffusion-LM (`train.py`)

Важно:
- у `train.py` по умолчанию включен профиль качества для ROC (`--apply_roc_profile True`);
- он автоматически подтягивает настройки ближе к оригиналу (e2e, более длинный anneal, большее latent-пространство, авто vocab size).

### 2.1 Основной запуск: смешанный датасет, обучение с нуля

```powershell
python scripts/train.py `
  --checkpoint_path diffusion_models/malbehav_all_qp `
  --model_arch transformer `
  --modality roc `
  --roc_train diffusion_lm/ROCstory `
  --batch_size 64 `
  --lr 0.0001 `
  --noise_schedule sqrt `
  --image_size 13 `
  --num_channels 128 `
  --num_res_blocks 2 `
  --dropout 0.1 `
  --seed 101 `
  --save_interval 100 `
  --log_interval 100 `
  --eval_interval 100 `
  --gradient_clipping 1.0 `
  --experiment random `
  --cache_mode yes `
  --apply_roc_profile True
```

Для чего:
- базовая и рекомендуемая тренировка для вашего MalBehav.

### 2.2 Тот же `train.py`, но только malware

```powershell
python scripts/train.py `
  --checkpoint_path diffusion_models/malbehav_malware_qp `
  --model_arch transformer `
  --modality roc `
  --roc_train diffusion_lm/ROCstory_malware `
  --batch_size 64 `
  --lr 0.0001 `
  --noise_schedule sqrt `
  --image_size 13 `
  --num_channels 128 `
  --num_res_blocks 2 `
  --dropout 0.15 `
  --seed 101 `
  --save_interval 100 `
  --log_interval 100 `
  --eval_interval 100 `
  --gradient_clipping 1.0 `
  --experiment random `
  --cache_mode yes `
  --apply_roc_profile True
```

Отличие от 2.1:
- меняется только источник данных (`ROCstory_malware`) и целевая папка чекпоинтов;
- `dropout` можно держать чуть выше/ниже по качеству на этом подклассе.

### 2.3 Тот же `train.py`, но только benign

```powershell
python scripts/train.py `
  --checkpoint_path diffusion_models/malbehav_benign_qp `
  --model_arch transformer `
  --modality roc `
  --roc_train diffusion_lm/ROCstory_benign `
  --batch_size 64 `
  --lr 0.0001 `
  --noise_schedule sqrt `
  --image_size 13 `
  --num_channels 128 `
  --num_res_blocks 2 `
  --dropout 0.2 `
  --seed 101 `
  --save_interval 100 `
  --log_interval 100 `
  --eval_interval 100 `
  --gradient_clipping 1.0 `
  --experiment random `
  --cache_mode yes `
  --apply_roc_profile True
```

Отличие от 2.2:
- benign-сплит + другая папка + другой `dropout`.

### 2.4 Продолжение обучения с чекпоинта (resume)

```powershell
Get-ChildItem diffusion_models/malbehav_all_qp/model*.pt
```

```powershell
python scripts/train.py `
  --checkpoint_path diffusion_models/malbehav_all_qp `
  --model_arch transformer `
  --modality roc `
  --roc_train diffusion_lm/ROCstory `
  --batch_size 64 `
  --lr 0.0001 `
  --noise_schedule sqrt `
  --image_size 13 `
  --num_channels 128 `
  --num_res_blocks 2 `
  --seed 101 `
  --resume_checkpoint diffusion_models/malbehav_all_qp/model000900.pt `
  --save_interval 100 `
  --log_interval 100 `
  --eval_interval 100 `
  --gradient_clipping 1.0 `
  --experiment random `
  --cache_mode yes `
  --apply_roc_profile True
```

Отличие от 2.1:
- добавляется `--resume_checkpoint`, обучение продолжается от существующего шага;
- профиль качества не ломает совместимость resume-конфига.

### 2.5 Legacy-режим (для строгого воспроизведения старых прогонов)

```powershell
python scripts/train.py `
  --checkpoint_path diffusion_models/malbehav_legacy_emb32 `
  --model_arch transformer `
  --modality roc `
  --roc_train diffusion_lm/ROCstory `
  --batch_size 64 `
  --lr 0.0001 `
  --noise_schedule sqrt `
  --diffusion_steps 1500 `
  --image_size 13 `
  --num_channels 128 `
  --in_channel 32 `
  --out_channel 32 `
  --training_mode emb `
  --predict_xstart True `
  --lr_anneal_steps 2500 `
  --weight_decay 0.01 `
  --num_res_blocks 2 `
  --dropout 0.15 `
  --seed 101 `
  --save_interval 100 `
  --log_interval 100 `
  --eval_interval 100 `
  --gradient_clipping 1.0 `
  --experiment random `
  --cache_mode yes `
  --apply_roc_profile False
```

Отличие от 2.1:
- отключен профиль качества;
- остаются старые гиперпараметры (`emb`, `in_channel=32`, короткий `lr_anneal_steps`).

## 3) Обучение AR-модели для метрик (`train_ar.py`)

### 3.1 AR-модель для смешанного датасета

```powershell
python scripts/train_ar.py `
  --roc_train diffusion_lm/ROCstory `
  --vocab_path diffusion_models/malbehav_all_qp/vocab.json `
  --output_dir ar_models/malbehav_all_ar `
  --epochs 50 `
  --batch_size 64 `
  --lr 3e-4 `
  --weight_decay 0.01 `
  --max_len 169 `
  --d_model 256 `
  --n_layers 4 `
  --n_heads 4 `
  --dropout 0.1 `
  --seed 101
```

### 3.2 AR-модель для benign-only

```powershell
python scripts/train_ar.py `
  --roc_train diffusion_lm/ROCstory_benign `
  --vocab_path diffusion_models/malbehav_benign_qp/vocab.json `
  --output_dir ar_models/malbehav_benign_ar `
  --epochs 50 `
  --batch_size 64 `
  --lr 3e-4 `
  --weight_decay 0.01 `
  --max_len 169 `
  --d_model 256 `
  --n_layers 4 `
  --n_heads 4 `
  --dropout 0.1 `
  --seed 101
```

Отличие 3.1 vs 3.2:
- датасет и словарь должны совпадать с diffusion-моделью, которую вы оцениваете.

## 4) Генерация (`text_sample.py`)

### 4.1 Быстрый базовый запуск (без тяжелых метрик)

```powershell
python scripts/text_sample.py `
  --model_path diffusion_models/malbehav_all_qp/ema_0.9999_000900.pt `
  --batch_size 64 `
  --num_samples 300 `
  --out_dir generation_outputs/all/quick `
  --top_p 0.95 `
  --verbose yes
```

Для чего:
- быстро посмотреть качество текстов/разнообразия.

### 4.2 Более жесткое округление/постобработка

```powershell
python scripts/text_sample.py `
  --model_path diffusion_models/malbehav_all_qp/ema_0.9999_000900.pt `
  --batch_size 64 `
  --num_samples 300 `
  --out_dir generation_outputs/all/rounding_strict `
  --top_p -1.0 `
  --rounding_start_t 300 `
  --rounding_topk 4 `
  --rounding_temperature 0.8 `
  --rounding_block_special_tokens True `
  --postprocess_generated_text True `
  --verbose yes
```

Отличие от 4.1:
- меняется стратегия дискретизации (`top_p` и rounding-параметры);
- обычно меньше мусорных токенов, но может падать разнообразие.

### 4.3 Полный прогон генерации + все метрики в одном запуске

```powershell
python scripts/text_sample.py `
  --model_path diffusion_models/malbehav_all_qp/ema_0.9999_000900.pt `
  --batch_size 64 `
  --num_samples 300 `
  --out_dir generation_outputs/all/full_metrics `
  --top_p 0.95 `
  --verbose yes `
  --compute_all_metrics True `
  --ar_model_path ar_models/malbehav_all_ar/best
```

Отличие от 4.1/4.2:
- дополнительно запускаются NLL(train/valid) и AR perplexity;
- дольше по времени, но дает полный отчет.

## 5) NLL-оценка (`nll.py`)

### 5.1 Valid NLL

```powershell
python scripts/nll.py `
  --model_path diffusion_models/malbehav_all_qp/ema_0.9999_000900.pt `
  --batch_size 64 `
  --num_samples 256 `
  --split valid `
  --out_dir scores `
  --clamp clamp
```

### 5.2 Train NLL

```powershell
python scripts/nll.py `
  --model_path diffusion_models/malbehav_all_qp/ema_0.9999_000900.pt `
  --batch_size 64 `
  --num_samples 256 `
  --split train `
  --out_dir scores `
  --clamp clamp
```

Отличие 5.1 vs 5.2:
- только `--split` (`valid` vs `train`) для проверки обобщения/оверфита.

### 5.3 Оценка на альтернативной ROC-папке

```powershell
python scripts/nll.py `
  --model_path diffusion_models/malbehav_all_qp/ema_0.9999_000900.pt `
  --batch_size 64 `
  --num_samples 256 `
  --split valid `
  --out_dir scores `
  --clamp clamp `
  --roc_train_override diffusion_lm/ROCstory_benign
```

Отличие от 5.1:
- принудительная подмена eval-корпуса через `--roc_train_override`.

### 5.4 Вариант без clamp

```powershell
python scripts/nll.py `
  --model_path diffusion_models/malbehav_all_qp/ema_0.9999_000900.pt `
  --batch_size 64 `
  --num_samples 256 `
  --split valid `
  --out_dir scores `
  --clamp noclamp
```

Отличие от 5.1:
- проверка чувствительности метрик к clamp/noclamp.

## 6) Сводные метрики отдельным скриптом (`run_all_metrics.py`)

```powershell
python scripts/run_all_metrics.py `
  --model_path diffusion_models/malbehav_all_qp/ema_0.9999_000900.pt `
  --generated_json generation_outputs/all/quick/malbehav_all_qp.ema_0.9999_000900.pt.samples_0.95.json `
  --ar_model_path ar_models/malbehav_all_ar/best `
  --modality roc `
  --experiment random `
  --nll_batch_size 64 `
  --nll_num_samples 256 `
  --nll_out_dir scores `
  --nll_clamp clamp `
  --out_json generation_outputs/all/quick/all_metrics.json
```

Для чего:
- когда уже есть `.json` с генерациями и нужно быстро собрать все метрики в один файл.

## 7) Прямая AR-perplexity оценка (`ppl_under_ar.py`)

```powershell
python scripts/ppl_under_ar.py `
  --model_path diffusion_models/malbehav_all_qp/ema_0.9999_000900.pt `
  --modality roc `
  --experiment random `
  --model_name_or_path ar_models/malbehav_all_ar/best `
  --input_text generation_outputs/all/quick/malbehav_all_qp.ema_0.9999_000900.pt.samples_0.95.json `
  --mode eval
```

Для чего:
- изолированная проверка только AR-score, без NLL.

## 8) Что смотреть после запусков

- Diffusion чекпоинты: `diffusion_models/<run_name>/model*.pt`, `ema_*.pt`.
- Аргументы обучения: `diffusion_models/<run_name>/training_args.json`.
- Генерации: `generation_outputs/.../*.txt`, `*.json`, `*.metrics.json`.
- NLL-скоры: `diffusion_models/<run_name>/ema_score_valid_nll.json`, `ema_score_train_nll.json`.
- AR-скоры: `diffusion_models/<run_name>/ema_score_decode.json`.

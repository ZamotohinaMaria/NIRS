# Diffusion-LM + MalbehavD-V1

## 1) Подготовка датасета

Из корня `Diffusion-LM-main`:

```powershell
python improved-diffusion/scripts/prepare_malbehav_data.py `
  --input_csv ../MalbehavD-V1-main/MalBehavD-V1-dataset.csv `
  --output_dir ../MalbehavD-V1-main/prepared_for_diffusionlm `
  --include_label_token yes `
  --use_all_for_train yes `
  --seed 101
```

Скрипт создаст:

- `malbehav_train.txt`
- `malbehav_valid.txt`
- `malbehav_test.txt`
- `malbehav_stats.json`

В `malbehav_stats.json` есть поле `estimated_vocab_size_threshold_gt10`. Используйте его в `--vocab_size`.
При `--use_all_for_train yes` весь датасет идет в train.

## 2) Обучение Diffusion-LM на MalbehavD-V1

Из папки `Diffusion-LM-main/improved-diffusion`:

```powershell
python scripts/run_train.py `
  --diff_steps 2000 `
  --save_interval 5000 `
  --checkpoint_root runs/models `
  --eval_interval 2000 `
  --eval_num_batches 4 `
  --early_stop_patience_eval 5 `
  --early_stop_min_delta 0.0 `
  --model_arch transformer `
  --lr 0.0001 `
  --lr_anneal_steps 300000 `
  --seed 101 `
  --noise_schedule sqrt `
  --in_channel 64 `
  --modality malbehav `
  --submit no `
  --padding_mode pad `
  --bsz 64 `
  --notes malbehav_xstart `
  --app "--predict_xstart True --training_mode e2e --vocab_size 250 --malbehav_train ../../MalbehavD-V1-main/prepared_for_diffusionlm/malbehav_train.txt --malbehav_valid ../../MalbehavD-V1-main/prepared_for_diffusionlm/malbehav_valid.txt --malbehav_test ../../MalbehavD-V1-main/prepared_for_diffusionlm/malbehav_test.txt --cache_mode no"
```

При желании можно заменить `--vocab_size 250` на актуальное значение из `malbehav_stats.json`, если вы измените параметры подготовки данных.
Чекпоинты будут сохраняться в `runs/models/<имя_запуска>/` каждые `save_interval` шагов, а лучшая модель по `eval_loss_mean` — в `best_model.pt`.
Если `eval_loss_mean` не улучшается 5 проверок подряд (`early_stop_patience_eval=5`), обучение автоматически остановится.
Перед запуском теперь выполняются preflight-проверки путей и прав записи; при ошибке отсутствующей папки/файла процесс завершится сразу, а не через часы обучения.
Логи дублируются в терминал и файл `runs/models/<имя_запуска>/console.log`. Табличные метрики пишутся в `log.txt/progress.csv` в той же папке.
Каждые `eval_interval` выводится строка `[EVAL] ...`, а прогресс обучения печатается каждые 10 шагов как `[PROGRESS] ...`.

## 3) Генерация `benign` и `malware` за один запуск

1. Найдите последний чекпоинт в папке `improved-diffusion/diffusion_models/...`.
2. Запустите:

```powershell
python scripts/generate_malbehav_by_label.py `
  --model_path diffusion_models/<ваша_папка_модели>/ema_0.9999_300000.pt `
  --target_label both `
  --num_samples 5000 `
  --samples_per_round 512 `
  --batch_size 64 `
  --top_p -1 `
  --max_rounds 50 `
  --out_dir generation_outputs
```

Скрипт остановится только когда наберет оба класса:
- `LABEL_0` (benign) в `malbehav_label0_*.txt`
- `LABEL_1` (malware) в `malbehav_label1_*.txt`
Лог генерации дублируется в `generation_outputs/generation_by_label_console.log`.

Если нужен один класс, можно указать `--target_label 0` или `--target_label 1`.

## 4) Конвертация сэмплов в CSV-формат MalbehavD-V1

```powershell
python scripts/malbehav_samples_to_csv.py `
  --input_txt generation_outputs/<samples>.txt `
  --output_csv generation_outputs/malbehav_generated.csv `
  --drop_special_tokens yes `
  --with_sha256_column yes
```

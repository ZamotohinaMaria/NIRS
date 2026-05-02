# Diffusion-LM + MalbehavD-V1

## 1) Подготовка датасета

Из корня `Diffusion-LM-main`:

```powershell
python improved-diffusion/scripts/prepare_malbehav_data.py `
  --input_csv ../MalbehavD-V1-main/MalBehavD-V1-dataset.csv `
  --output_dir ../MalbehavD-V1-main/prepared_for_diffusionlm `
  --include_label_token yes `
  --train_ratio 0.9 `
  --valid_ratio 0.05 `
  --seed 101
```

Скрипт создаст:

- `malbehav_train.txt`
- `malbehav_valid.txt`
- `malbehav_test.txt`
- `malbehav_stats.json`

В `malbehav_stats.json` есть поле `estimated_vocab_size_threshold_gt10`. Используйте его в `--vocab_size`.

## 2) Обучение Diffusion-LM на MalbehavD-V1

Из папки `Diffusion-LM-main/improved-diffusion`:

```powershell
python scripts/run_train.py `
  --diff_steps 2000 `
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

## 3) Генерация новых последовательностей API

1. Найдите последний чекпоинт в папке `improved-diffusion/diffusion_models/...`.
2. Запустите:

```powershell
python scripts/text_sample.py `
  --model_path diffusion_models/<ваша_папка_модели>/ema_0.9999_300000.pt `
  --batch_size 64 `
  --num_samples 5000 `
  --top_p -1 `
  --out_dir generation_outputs `
  --verbose yes
```

Текстовые сэмплы будут в `generation_outputs/*.txt`.

## 4) Конвертация сэмплов в CSV-формат MalbehavD-V1

```powershell
python scripts/malbehav_samples_to_csv.py `
  --input_txt generation_outputs/<samples>.txt `
  --output_csv generation_outputs/malbehav_generated.csv `
  --drop_special_tokens yes `
  --with_sha256_column yes
```

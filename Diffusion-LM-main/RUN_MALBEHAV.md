# Diffusion-LM Minimal Run For MalbehavD-V1

## 0) Install local package
Run from `Diffusion-LM-main`:

```powershell
pip install -e .
```

## 1) Prepare dataset in ROC format
Run from `Diffusion-LM-main`:

```powershell
python scripts/prepare_malbehav_roc.py `
  --input_csv ../MalbehavD-V1-main/MalBehavD-V1-dataset.csv `
  --out_dir diffusion_lm/ROCstory `
  --valid_size 500 `
  --label_filter all
```

Notes:
- `--label_filter malware` keeps only label `1`.
- `--label_filter benign` keeps only label `0`.

## 2) Train diffusion model (text/API generation)
Run from `Diffusion-LM-main`:

```powershell
python scripts/train.py `
  --checkpoint_path diffusion_models/malbehav_roc_rand16 `
  --model_arch transformer `
  --modality roc `
  --save_interval 50000 `
  --lr 0.0001 `
  --batch_size 64 `
  --diffusion_steps 2000 `
  --noise_schedule sqrt `
  --use_kl False `
  --learn_sigma False `
  --image_size 8 `
  --num_channels 128 `
  --seed 101 `
  --dropout 0.1 `
  --in_channel 16 `
  --out_channel 16 `
  --padding_mode pad `
  --experiment random `
  --lr_anneal_steps 400000 `
  --weight_decay 0.0 `
  --num_res_blocks 2 `
  --predict_xstart True `
  --training_mode emb `
  --roc_train diffusion_lm/ROCstory
```

## 3) Generate samples
Replace checkpoint path with your real checkpoint (`model*.pt` or `ema_*.pt`):

```powershell
python scripts/text_sample.py `
  --model_path diffusion_models/malbehav_roc_rand16/ema_0.9999_50000.pt `
  --batch_size 50 `
  --num_samples 500 `
  --out_dir generation_outputs `
  --top_p -1.0 `
  --verbose yes
```

## 4) Compute NLL metrics

```powershell
python scripts/nll.py `
  --model_path diffusion_models/malbehav_roc_rand16/ema_0.9999_50000.pt `
  --batch_size 64 `
  --num_samples 256 `
  --split valid `
  --out_dir scores `
  --clamp clamp
```

Important:
- `nll.py` expects dataset at `diffusion_lm/ROCstory` (hardcoded in script logic), so keep that output path from step 1.

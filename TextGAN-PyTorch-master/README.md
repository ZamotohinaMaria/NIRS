# TextGAN-PyTorch (RelGAN-only, tuned for MalbehavD-V1)

This repository was cleaned to keep only the parts needed for **RelGAN** on real data.

## Included workflow

1. Prepare `MalBehavD-V1-dataset.csv` into text sequences for RelGAN.
2. Train two dedicated models for higher fidelity:
- `goodware` model (label `0`)
- `malware` model (label `1`)
3. Generate synthetic sequences.
4. Export generated sequences back to **the same CSV schema** as MalbehavD-V1.

## Requirements

- Python 3.6+
- PyTorch >= 1.0.0
- `pip install -r requirements.txt`

## 1) Prepare dataset from CSV

Run from repository root:

```bash
python tools/prepare_malbehavd_relgan.py ^
  --input_csv ..\MalbehavD-V1-main\MalBehavD-V1-dataset.csv ^
  --output_root dataset ^
  --dataset_prefix MalbehavD-V1 ^
  --test_ratio 0.2 ^
  --seed 42
```

Created files:

- `dataset/MalbehavD-V1_goodware.txt`
- `dataset/testdata/MalbehavD-V1_goodware_test.txt`
- `dataset/MalbehavD-V1_malware.txt`
- `dataset/testdata/MalbehavD-V1_malware_test.txt`

## 2) Train RelGAN (high-fidelity preset)

From `run/`:

```bash
cd run
python run_relgan.py 0 0  # goodware
python run_relgan.py 1 0  # malware
```

`run_relgan.py` now uses a high-fidelity preset for this dataset (longer MLE pretrain, class-specific training, conservative adversarial LR).

## 3) Generate synthetic sequences

Example for malware model checkpoint:

```bash
python tools/generate_relgan_samples.py ^
  --dataset MalbehavD-V1_malware ^
  --checkpoint save\...\models\gen_ADV_XXXXX.pt ^
  --output_txt generated\malware_samples.txt ^
  --num_samples 1285 ^
  --batch_size 64
```

Do the same for goodware with `--dataset MalbehavD-V1_goodware`.

## 4) Convert generated samples back to MalbehavD format

Export class files:

```bash
python tools/export_relgan_samples_to_csv.py ^
  --samples_txt generated\goodware_samples.txt ^
  --template_csv ..\MalbehavD-V1-main\MalBehavD-V1-dataset.csv ^
  --output_csv generated\goodware_generated.csv ^
  --label 0

python tools/export_relgan_samples_to_csv.py ^
  --samples_txt generated\malware_samples.txt ^
  --template_csv ..\MalbehavD-V1-main\MalBehavD-V1-dataset.csv ^
  --output_csv generated\malware_generated.csv ^
  --label 1
```

Merge into one CSV:

```bash
python tools/merge_generated_csv.py ^
  --goodware_csv generated\goodware_generated.csv ^
  --malware_csv generated\malware_generated.csv ^
  --output_csv generated\MalBehavD-V1-generated.csv
```

The output keeps the original schema: `sha256,labels,0,1,...`.

## Notes

- Only RelGAN and real-data mode are supported.
- `utils/text_process.py` was adjusted to preserve API token case (no lowercase conversion).
- `run_signal.txt` still supports early-stop control (`pre_sig`, `adv_sig`).

# COAST: Context-Aware Differential Learning for Gene Expression Prediction in Spatial Transcriptomics

[![Paper](https://img.shields.io/badge/Paper-MICCAI%202026-blue)](https://link.springer.com/chapter/10.1007/978-3-032-38224-5_6)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](#installation)

Official implementation of COAST: Context-Aware Differential Learning for Gene Expression Prediction in Spatial Transcriptomics


## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
or

```bash
conda env create -f environment.yml
conda activate COAST
```

## Configuration
Set the cache and output paths in `configs/baseline.yaml`.

The default experiment uses:
```text
configs/datasets/main
configs/models/main
configs/losses/main
```

Extended dataset, loss, and ablation configurations are stored under the corresponding `all` and `ablations` directories.

## Single experiment

```bash
python -m scripts.train \
  --base_config configs/baseline.yaml \
  --data_config configs/datasets/main/abalo_human_squamous_cell_carcinoma.yaml \
  --model_config configs/models/main/COAST.yaml \
  --loss_config configs/losses/main/0_N18F72LR002S1D1.yaml
```

## Multi-GPU experiments

```bash
bash scripts/train_all.sh
```

The launcher detects the number of visible GPUs. Directories can be overridden without editing the script:

```bash
NUM_GPUS=8 \
DATA_CONFIG_DIR=configs/datasets/all \
MODEL_CONFIG_DIR=configs/models/ablations \
LOSS_CONFIG_DIR=configs/losses/all \
bash scripts/train_all.sh
```

The launcher stops when an experiment fails and reports the corresponding log file.

## External inference

```bash
python -m scripts.evaluate_external --help
```

```bash
python -m torch.distributed.run --nproc_per_node=8 \
  -m scripts.evaluate_external_ddp --help
```

## Preprocessing

```bash
python -m scripts.preprocessing.extract_coordinates --help
python -m scripts.preprocessing.export_slide_info --help
python -m scripts.preprocessing.cache_spared --help
python -m scripts.preprocessing.cache_external_knn --help
```

## Analysis

```bash
python -m scripts.analysis.compute_group_pcc --help
python -m scripts.analysis.run_group_pcc --help
python -m scripts.analysis.compute_p_value --help
python -m scripts.analysis.create_gene_groups --help
python -m scripts.analysis.summarize_logs --help
python -m scripts.analysis.visualize_spots --help
python -m scripts.analysis.visualize_external_predictions --help
```

---

## License

This repository is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
Commercial use is not permitted. Non-commercial research and educational use is permitted with appropriate attribution.  
This license applies only to the original COAST source code in this repository. Third-party resources are subject to their own licenses and usage restrictions.  

---

## Citation

If you find this repository useful, please cite COAST:

@inproceedings{byeon2026coast,
  title={COAST: Context-Aware Differential Learning for Gene Expression Prediction in Spatial Transcriptomics},
  author={Byeon, Keunho and Park, Sunhong and Lim, Jeewoo and Kwak, Jin Tae},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={55--65},
  year={2026},
  organization={Springer}
}

---

## Acknowledgements

This work was supported by a grant of the National Research
Foundation of Korea (NRF) (No. RS-2025-00558322 and RS-2024-00397293).

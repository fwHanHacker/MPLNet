# MPLNet: Multi-Morphology Power Line Instance Detection Network

## Repository Structure

```text
MPLNet/
|-- configs/mfplnet/        # Experiment configuration files
|-- docs/                   # Dataset
|-- mfplnet/                # Core package
|   |-- datasets/           # Dataset readers and preprocessing pipeline
|   |-- engine/             # Runner, optimizer, scheduler, recorder
|   |-- models/             # Backbones, necks, heads, losses, utilities
|   |-- ops/                # NMS operator and fallback implementation
|   `-- utils/              # Config, metrics, logging, visualization
|-- tools/                  # Dataset conversion and segmentation tools
|-- main.py                 # Training / validation / testing entry point
|-- requirements.txt        # Python dependencies
`-- setup.py                # Package installation script
```

## Installation

```bash
cd MPLNet

conda create -n mplnet python=3.8 -y
conda activate mplnet

pip install -r requirements.txt
pip install -e .
```

## Data Preparation

Place datasets under `mfplnet/data/` by default:

```text
mfplnet/data/
|-- cpld_01/
|-- pldu_01/
`-- pldm_01/
```

Each dataset split follows a TuSimple-style annotation format with JSON-lines files such as `label_data_0114.json`, image paths under `clips/`, and segmentation masks under `seg_label/`.

See [docs/DATASET.md](docs/DATASET.md) for the expected layout and conversion notes.

## Training

Train MPLNet with a selected config:

```bash
python main.py configs/mfplnet/mfpl_resnet18_cpld_01.py --gpus 0
```

Resume or fine-tune from a checkpoint:

```bash
python main.py configs/mfplnet/mfpl_resnet18_cpld_01.py --gpus 0 --resume_from work_dirs/path/to/checkpoint.pth
python main.py configs/mfplnet/mfpl_resnet18_cpld_01.py --gpus 0 --finetune_from path/to/pretrained.pth
```

## Evaluation

Validate or test a checkpoint:

```bash
python main.py configs/mfplnet/mfpl_resnet18_cpld_01.py --gpus 0 --validate --load_from path/to/checkpoint.pth
python main.py configs/mfplnet/mfpl_resnet18_cpld_01.py --gpus 0 --test --load_from path/to/checkpoint.pth
```

Enable qualitative visualization:

```bash
python main.py configs/mfplnet/mfpl_resnet18_cpld_01.py --gpus 0 --test --view --load_from path/to/checkpoint.pth
```

Outputs are saved to the configured `work_dirs/` directory.

## Configs

| Config | Dataset | Backbone |
| --- | --- | --- |
| `mfpl_resnet18_cpld_01.py` | CPLD | ResNet-18 |
| `mfpl_resnet34_cpld_01.py` | CPLD | ResNet-34 |
| `mfpl_hrnet_cpld_01.py` | CPLD | HRNet |
| `mfpl_efficientvit_m0_cpld_01.py` | CPLD | EfficientViT-M0 |
| `mfpl_mit_efficientvit_l0_cpld_01.py` | CPLD | MIT-EfficientViT-L0 |
| `mfpl_resnet18_pldu_01.py` | PLDU | ResNet-18 |
| `mfpl_hrnet_pldu_01.py` | PLDU | HRNet |
| `mfpl_resnet18_pldm_01.py` | PLDM | ResNet-18 |

## License

This project is released under the [MIT License](LICENSE).

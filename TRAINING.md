# Training Instructions

## Dataset

The project expects an MRI dataset zip file with folders such as:

- `Training/glioma`
- `Training/meningioma`
- `Training/notumor`
- `Training/pituitary`
- `Testing/glioma`
- `Testing/meningioma`
- `Testing/notumor`
- `Testing/pituitary`

## Run training

```powershell
python model/train.py --dataset_zip "C:\Users\safsa\Downloads\archive.zip" --epochs 12 --batch_size 16
```

## What the pipeline does

- Extracts the zip archive
- Validates image files and skips corrupted files
- Organizes classes into a normalized layout
- Creates train, validation, and test splits
- Trains:
  - `VGG16`
  - `ResNet50`
  - `EfficientNetB0`
  - `MobileNetV2`
- Uses:
  - early stopping
  - ReduceLROnPlateau
  - checkpoint saving
  - CSV logs
  - TensorBoard
  - mixed precision when GPU is available
- Saves the best model to `models/exports/best_model.keras`

## Training outputs

- `logs/training_summary.json`
- `logs/*_metrics.json`
- `logs/*_training.csv`
- `tensorboard/`
- `data/artifacts/plots/`
- `models/checkpoints/`
- `models/exports/`

## TensorBoard

```powershell
tensorboard --logdir tensorboard
```

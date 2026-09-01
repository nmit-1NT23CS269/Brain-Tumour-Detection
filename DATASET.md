# Brain Tumor MRI Dataset Instructions

NeuroScan AI requires the standard **Brain Tumor MRI Dataset** from Kaggle for transfer learning model training and dataset-level analytics.

## 1. Dataset Source

Download the dataset from Kaggle:
- **Kaggle Dataset**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## 2. Directory Layout Requirement

Extract the downloaded dataset ZIP file into `data/raw/archive/` within the repository root so that it follows this structure:

```text
data/raw/archive/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## 3. Class Names Mapping

The model expects the following 4 classes:
1. `glioma` (Glioma Tumour)
2. `meningioma` (Meningioma Tumour)
3. `notumor` (Normal / No Tumour)
4. `pituitary` (Pituitary Tumour)

## 4. Verification

After extracting the dataset, verify the layout by running:

```powershell
python -c "from pathlib import Path; print('Training classes:', [p.name for p in Path('data/raw/archive/Training').iterdir() if p.is_dir()])"
```

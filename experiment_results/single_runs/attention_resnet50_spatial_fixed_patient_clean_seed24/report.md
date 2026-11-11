# SMC-LUD spatially corrected attention ResNet-50, patient-disjoint Clean data

- Model: model.py CustomModelSpatialFixed, ImageNet-pretrained ResNet-50 backbone, two 7x7 attention branches, spatial attention, two classes
- Original CustomModel is preserved; this model applies channel weights to 7x7 features before spatial attention.
- The ordinary ResNet-50 baseline was randomly initialized, so differences are not a pure attention ablation.
- Data: Clean images, new patient-disjoint split, seed 24
- Selection: minimum validation loss; test evaluated after selection
- Positive class: HCC (ImageFolder index 0), threshold 0.5
- Patient prediction: arithmetic mean of image-level HCC probabilities
- This split differs from the published image allocation; results are not a direct replication of Table 3.
- Hem_87 lacks a patient-table row; the main test retains its folder label and sensitivity excludes it.

## Run

- Epochs completed: 19
- Best epoch: 12
- Actual batch size: 32

## Test results

| Metric | Image level | Patient level |
|---|---:|---:|
| accuracy | 0.9921 | 1.0000 |
| balanced_accuracy | 0.9915 | 1.0000 |
| hcc_precision | 0.9856 | 1.0000 |
| hcc_sensitivity_recall | 1.0000 | 1.0000 |
| hemangioma_specificity | 0.9829 | 1.0000 |
| hcc_f1 | 0.9928 | 1.0000 |
| macro_f1 | 0.9921 | 1.0000 |
| roc_auc_hcc | 1.0000 | 1.0000 |
| brier_hcc | 0.0044 | 0.0024 |

## Hem_87 sensitivity analysis

Excluding 5 Clean images from one test patient:

- Image accuracy: 0.9920
- Patient accuracy: 1.0000

## Patient bootstrap intervals

Patients resampled with replacement; 1000 replicates requested.

- accuracy: [1.0000, 1.0000] (1000 valid resamples)
- hcc_precision: [1.0000, 1.0000] (1000 valid resamples)
- hcc_sensitivity_recall: [1.0000, 1.0000] (1000 valid resamples)
- hemangioma_specificity: [1.0000, 1.0000] (1000 valid resamples)
- hcc_f1: [1.0000, 1.0000] (1000 valid resamples)
- roc_auc_hcc: [1.0000, 1.0000] (1000 valid resamples)

See run_config.json, history.csv, predictions CSVs, metrics JSONs, and figures for details.

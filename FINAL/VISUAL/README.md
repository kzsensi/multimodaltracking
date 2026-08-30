# Final Visual FER Report

Final selected visual-only facial emotion recognition model:

```text
models/torch/efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue_v2.pt
```

## Final Metrics

Real-time webcam mode:

```text
Accuracy: 85.22%
Macro F1: 84.86%
```

Offline TTA evaluation:

```text
Accuracy: 86.65%
Macro F1: 86.30%
```

The real-time number is the number to use for webcam inference. The TTA number
is useful for reports because it averages original and horizontally flipped
predictions, but it is slower.

## Weak-Class F1 Improvement

The weak classes were sad, fear, angry, and neutral.

Real-time F1:

```text
sad:     81.82%
fear:    83.37%
angry:   83.09%
neutral: 82.64%
```

TTA F1:

```text
sad:     82.63%
fear:    84.79%
angry:   85.50%
neutral: 84.71%
```

## Included Graphs

Required graphs:

```text
realtime_test_auc_curve.png
realtime_test_confusion_matrix.png
realtime_class_distribution_bar_chart.png
realtime_efficiency_graph.png
```

Clean final graph names:

```text
01_AUC_CURVE_GRAPH.png
02_CONFUSION_MATRIX_GRAPH.png
03_BAR_CHART_CLASS_DISTRIBUTION.png
04_EFFICIENCY_GRAPH.png
```

Research/report-friendly vector exports are also included:

```text
01_AUC_CURVE_GRAPH.pdf
01_AUC_CURVE_GRAPH.svg
02_CONFUSION_MATRIX_GRAPH.pdf
02_CONFUSION_MATRIX_GRAPH.svg
03_BAR_CHART_CLASS_DISTRIBUTION.pdf
03_BAR_CHART_CLASS_DISTRIBUTION.svg
04_EFFICIENCY_GRAPH.pdf
04_EFFICIENCY_GRAPH.svg
```

Use PNG for Word/PowerPoint and quick submission. Use PDF for LaTeX, printed
reports, or thesis documents. Use SVG when you need to edit the graph in tools
such as Figma, Illustrator, Inkscape, or a browser-based editor.

Extra useful graphs:

```text
realtime_training_curves.png
realtime_validation_auc_curve.png
realtime_validation_confusion_matrix.png
model_accuracy_comparison_bar_chart.png
weak_class_f1_comparison_bar_chart.png
tta_test_auc_curve.png
tta_test_confusion_matrix.png
tta_efficiency_graph.png
```

## Training Recipe

Base training:

```text
Architecture: EfficientNet-B1
Training data: dataset_yolo_emotions_balanced_1k/train + dataset_mutfer2024/train
Validation: dataset_mutfer2024/validation
Test: dataset_mutfer2024/test
Epochs: 7
Batch size: 16
Learning rate: 0.0001
Augmentation: light
```

Continuation training:

```text
Initial checkpoint: efficientnet_b1_yolo1k_plus_mutfer_finetune.pt
Epochs: 4
Learning rate: 0.00003
Augmentation: rich
Label smoothing: 0.03
Weak-class loss boost: sad=1.20 fear=1.20 angry=1.15 neutral=1.10
```

Final polishing run:

```text
Initial checkpoint: efficientnet_b1_yolo1k_plus_mutfer_weakboost_continue.pt
Epochs: 3
Learning rate: 0.00001
Augmentation: rich
Label smoothing: 0.03
Weak-class loss boost: sad=1.15 fear=1.15 angry=1.10 neutral=1.08
```

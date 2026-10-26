# Shared revision metrics

The unchanged implementation is tools/evaluate_model.py; metrics/revision.py re-exports its functions. No metric formula or parameter is altered.

Image AUROC uses sklearn.metrics.roc_auc_score on image labels and each final map maximum, times 100. Pixel AUROC uses the same function on every flattened test pixel, including complete normal and anomaly images. Normal GT is a zero native-size mask. Anomaly GT is official, converted to grayscale, nearest-neighbor resized only when dimensions differ from RGB, then binarized with >0.

Preprocessing is full-view 256 RGB/ImageNet normalization. A 56x56 fused response is padded by 4 per edge to 64x64 and bilinearly resized to original image dimensions (align_corners=False default). Metrics use complete native image/GT domains, not a 256 crop or anomaly-only pixels. There is no test min-max, tuned threshold, score recentering or region filtering.

AU-PRO retains exact score ties, 8-connected GT components,equal per-region PRO weights, and the entire test background FPR denominator. The FPR cap is 0.3, with linear interpolation at the cap and trapz over FPR/0.3, times 100. Threshold sweeping belongs to ROC/PRO calculation; it is not test tuning of a deployed threshold.

Teacher statistics use optimization training only; q0.90/q0.995 use normal validation only. Final evaluation uses explicit 70000-step last bundles. Dataset macro is category-equal arithmetic mean; partial categories and mixed variants/splits are rejected. Calibration recomputation may differ numerically across environments; exact identity to historical archived-map calibration is not asserted.

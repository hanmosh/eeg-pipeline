# 256 Augmentation Before vs After

Weighted / 2D CNNGRU on 256 scalograms.

| Run | EEG Acc | EEG F1 | EEG AUC | Survey Acc | Survey F1 | Survey AUC | Fusion Acc | Fusion F1 | Fusion AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Before augmentation | 0.6250 | 0.6022 | 0.6500 | 0.8000 | 0.7925 | 0.9500 | 0.8750 | 0.8770 | 0.9000 |
| After best EEG augmentation | 0.7000 | 0.6548 | 0.6875 | 0.8000 | 0.7925 | 0.9500 | 0.8750 | 0.8770 | 0.9000 |

Best EEG augmentation used:

- SpecAugment: `p=0.1`, `freq_mask_param=5`, `time_mask_param=10`, `num_freq_masks=1`, `num_time_masks=1`, `mask_value=0.0`
- Gaussian noise: `p=0.1`, `std=0.01`, `clamp_min=0.0`, `clamp_max=1.0`

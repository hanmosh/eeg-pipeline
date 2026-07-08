# Final Result Tables

Values are taken directly from the logged `cv_avg_test_*` metrics.

## RAW 1D EEG Models
| Label | Model | EEG Acc | EEG Prec | EEG Rec | EEG F1 | EEG AUC | Survey Acc | Survey Prec | Survey Rec | Survey F1 | Survey AUC | Fusion Acc | Fusion Prec | Fusion Rec | Fusion F1 | Fusion AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Specific | 1D RawChronoNet | 0.5000 | 0.4810 | 0.4000 | 0.3826 | 0.3750 | 0.8250 | 0.8333 | 0.8000 | 0.8143 | 0.8875 | 0.7500 | 0.8500 | 0.6500 | 0.7014 | 0.8500 |
| Specific | 1D RawCNNGRU | 0.4000 | 0.2190 | 0.2500 | 0.2234 | 0.4250 | 0.8250 | 0.8333 | 0.8000 | 0.8143 | 0.8875 | 0.8500 | 0.9000 | 0.8000 | 0.8429 | 0.8625 |
| Specific | 1D RawCNNLSTM | 0.5750 | 0.6143 | 0.5000 | 0.4388 | 0.5000 | 0.8250 | 0.8333 | 0.8000 | 0.8143 | 0.8875 | 0.8500 | 0.8500 | 0.8500 | 0.8500 | 0.9000 |
| Composite | 1D RawChronoNet | 0.5250 | 0.5033 | 0.4000 | 0.4310 | 0.4875 | 0.7750 | 0.7843 | 0.8500 | 0.8002 | 0.8625 | 0.7250 | 0.7343 | 0.8000 | 0.7502 | 0.8500 |
| Composite | 1D RawCNNGRU | 0.4500 | 0.3500 | 0.4000 | 0.3500 | 0.3875 | 0.7750 | 0.7843 | 0.8500 | 0.8002 | 0.8625 | 0.8000 | 0.8033 | 0.8500 | 0.8148 | 0.8250 |
| Composite | 1D RawCNNLSTM | 0.5500 | 0.3500 | 0.5500 | 0.4167 | 0.4625 | 0.7750 | 0.7843 | 0.8500 | 0.8002 | 0.8625 | 0.8000 | 0.8033 | 0.8500 | 0.8148 | 0.8500 |
| Factor | 1D RawChronoNet | 0.5250 | 0.3333 | 0.4667 | 0.3810 | 0.4575 | 0.8000 | 0.7700 | 0.7500 | 0.7448 | 0.8417 | 0.8750 | 0.8833 | 0.8167 | 0.8362 | 0.8158 |
| Factor | 1D RawCNNGRU | 0.4500 | 0.3476 | 0.6000 | 0.4343 | 0.4850 | 0.8000 | 0.7700 | 0.7500 | 0.7448 | 0.8417 | 0.8000 | 0.7700 | 0.7500 | 0.7448 | 0.8417 |
| Factor | 1D RawCNNLSTM | 0.4500 | 0.4893 | 0.6667 | 0.4879 | 0.3925 | 0.8000 | 0.7700 | 0.7500 | 0.7448 | 0.8417 | 0.8000 | 0.7700 | 0.7500 | 0.7448 | 0.8283 |
| Weighted | 1D RawChronoNet | 0.4500 | 0.2000 | 0.1500 | 0.1667 | 0.3000 | 0.8000 | 0.8433 | 0.8000 | 0.7925 | 0.9500 | 0.8000 | 0.8933 | 0.7500 | 0.7759 | 0.8875 |
| Weighted | 1D RawCNNGRU | 0.6250 | 0.7500 | 0.3500 | 0.4433 | 0.6125 | 0.8000 | 0.8433 | 0.8000 | 0.7925 | 0.9500 | 0.8250 | 0.8433 | 0.8500 | 0.8306 | 0.9250 |
| Weighted | 1D RawCNNLSTM | 0.5500 | 0.3500 | 0.4500 | 0.3833 | 0.7125 | 0.8000 | 0.8433 | 0.8000 | 0.7925 | 0.9500 | 0.8500 | 0.8433 | 0.9000 | 0.8592 | 0.9500 |

## SCALOGRAM 2D Models
| Label | Model | EEG Acc | EEG Prec | EEG Rec | EEG F1 | EEG AUC | Survey Acc | Survey Prec | Survey Rec | Survey F1 | Survey AUC | Fusion Acc | Fusion Prec | Fusion Rec | Fusion F1 | Fusion AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Specific | 2D ChronoNet | 0.4750 | 0.4133 | 0.6500 | 0.5034 | 0.3250 | 0.8250 | 0.8333 | 0.8000 | 0.8143 | 0.8875 | 0.8250 | 0.8333 | 0.8000 | 0.8143 | 0.8000 |
| Specific | 2D CNNGRU | 0.4500 | 0.4310 | 0.6000 | 0.4926 | 0.5375 | 0.8250 | 0.8333 | 0.8000 | 0.8143 | 0.8875 | 0.8250 | 0.8500 | 0.8000 | 0.8214 | 0.8875 |
| Specific | 2D CNNLSTM | 0.5250 | 0.5167 | 0.8000 | 0.6071 | 0.5500 | 0.8250 | 0.8333 | 0.8000 | 0.8143 | 0.8875 | 0.8500 | 0.8500 | 0.8500 | 0.8500 | 0.8875 |
| Composite | 2D ChronoNet | 0.5750 | 0.5257 | 0.5500 | 0.5091 | 0.5500 | 0.7750 | 0.7843 | 0.8500 | 0.8002 | 0.8625 | 0.7750 | 0.7733 | 0.8500 | 0.7981 | 0.8500 |
| Composite | 2D CNNGRU | 0.6000 | 0.6167 | 0.4500 | 0.5119 | 0.5625 | 0.7750 | 0.7843 | 0.8500 | 0.8002 | 0.8625 | 0.7500 | 0.7733 | 0.8000 | 0.7600 | 0.8375 |
| Composite | 2D CNNLSTM | 0.5250 | 0.4333 | 0.5500 | 0.4476 | 0.5750 | 0.7750 | 0.7843 | 0.8500 | 0.8002 | 0.8625 | 0.7750 | 0.7843 | 0.8500 | 0.8002 | 0.8625 |
| Factor | 2D ChronoNet | 0.4250 | 0.2857 | 0.6500 | 0.3882 | 0.3817 | 0.8000 | 0.7700 | 0.7500 | 0.7448 | 0.8417 | 0.8000 | 0.7833 | 0.7500 | 0.7262 | 0.8150 |
| Factor | 2D CNNGRU | 0.5750 | 0.5600 | 0.5167 | 0.4800 | 0.5017 | 0.8000 | 0.7700 | 0.7500 | 0.7448 | 0.8417 | 0.8500 | 0.8333 | 0.8167 | 0.8076 | 0.8017 |
| Factor | 2D CNNLSTM | 0.4000 | 0.3000 | 0.5167 | 0.3482 | 0.2633 | 0.8000 | 0.7700 | 0.7500 | 0.7448 | 0.8417 | 0.8000 | 0.7700 | 0.7500 | 0.7448 | 0.8283 |
| Weighted | 2D ChronoNet | 0.5500 | 0.2476 | 0.4000 | 0.3055 | 0.4625 | 0.8000 | 0.8433 | 0.8000 | 0.7925 | 0.9500 | 0.8250 | 0.8933 | 0.7500 | 0.8063 | 0.8750 |
| Weighted | 2D CNNGRU | 0.5750 | 0.6143 | 0.6500 | 0.6010 | 0.5750 | 0.8000 | 0.8433 | 0.8000 | 0.7925 | 0.9500 | 0.8750 | 0.8433 | 0.9500 | 0.8878 | 0.9250 |
| Weighted | 2D CNNLSTM | 0.4750 | 0.3800 | 0.5500 | 0.4222 | 0.4125 | 0.8000 | 0.8433 | 0.8000 | 0.7925 | 0.9500 | 0.8250 | 0.8433 | 0.8500 | 0.8211 | 0.9375 |

Commands used:

```powershell
$configs = @(
  '.\run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_tfrecord_best_training.json'
  '.\run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_cnn_gru_tfrecord_best_training.json'
  '.\run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_cnn_lstm_tfrecord_best_training.json'
  '.\run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_raw_chrononet_csv_best_training.json'
  '.\run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_switchable_best_training.json'
  '.\run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_raw_cnn_lstm_csv_best_training.json'
)

foreach ($cfg in $configs) {
  .\.venv\Scripts\python .\pipeline.py $cfg -m
}
```

Configs used:

- `run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_tfrecord_best_training.json`
- `run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_cnn_gru_tfrecord_best_training.json`
- `run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_cnn_lstm_tfrecord_best_training.json`
- `run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_raw_chrononet_csv_best_training.json`
- `run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_switchable_best_training.json`
- `run_configs\grid_search_best_training_coarse_grid_3\belonging_config_multimodal_raw_cnn_lstm_csv_best_training.json`
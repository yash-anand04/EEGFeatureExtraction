# Phase 6 Analysis Report

## Benchmark Cross-Comparison

| method                          |    rmse |
|:--------------------------------|--------:|
| hybrid_residual_mlp             | 2.23678 |
| ridge_regression                | 2.2443  |
| hybrid_residual_ridge           | 2.2443  |
| distance_weighted_interpolation | 3.82538 |

## Best Method Per Channel Count

|   n_input_channels | method              |    rmse |
|-------------------:|:--------------------|--------:|
|                  5 | hybrid_residual_mlp | 2.49402 |
|                 10 | hybrid_residual_mlp | 1.96249 |
|                 15 | hybrid_residual_mlp | 2.25381 |

## Insights Generation

- Minimum channel count meeting RMSE <= 2.5: 5 using hybrid_residual_mlp (RMSE=2.4940)
- Best-preserved band (lowest mean RMSE): gamma; Most difficult band (highest mean RMSE): alpha
- Most critical missing electrodes from global ablation:
| removed_channel   |    rmse |   pearson_r |
|:------------------|--------:|------------:|
| POz               | 6.29085 |    0.834735 |
| Fz                | 5.32137 |    0.880283 |
| C5                | 5.30858 |    0.842528 |
| C6                | 5.11375 |    0.869999 |
| CP3               | 4.3643  |    0.907406 |

## Latency Analysis

| method                          |   train_seconds |   predict_seconds |   predict_ms_per_sample |
|:--------------------------------|----------------:|------------------:|------------------------:|
| ridge_regression                |       0.0052675 |       0.0005742   |             4.785e-05   |
| hybrid_residual_ridge           |       0.042352  |       0.000866833 |             7.22361e-05 |
| distance_weighted_interpolation |       0         |       0.0011341   |             9.45083e-05 |
| mlp_regression                  |      15.2212    |       0.00765597  |             0.000637997 |

## Practical Conclusions

- Hybrid residual models can be directly compared against interpolation baselines with unified signal, spatial, and spectral metrics.
- Electrode ablation exposes location-specific failure points that should guide wearable channel placement design.
- Latency trends are now included to support real-time feasibility discussion.

## Publication Visual Artifacts

- e:/Github/EEGFeatureExtraction/processed/bci_competition_iv_2a/phase5_publication_plots/timeseries_overlay.png
- e:/Github/EEGFeatureExtraction/processed/bci_competition_iv_2a/phase5_publication_plots/topomap_error_maps.png
- e:/Github/EEGFeatureExtraction/processed/bci_competition_iv_2a/phase5_publication_plots/psd_comparison.png
- e:/Github/EEGFeatureExtraction/processed/bci_competition_iv_2a/phase5_publication_plots/electrode_error_topology.png

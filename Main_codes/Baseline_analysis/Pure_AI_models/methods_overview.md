# Pure AI Methods Overview

This folder is now organized as one method per subfolder.

## Implemented Methods

- `deep_autoencoder`: fully connected encoder-decoder baseline
- `deep_cnn`: 1D convolutional regressor baseline
- `lstm`: recurrent sequence model over channel tokens
- `tcn`: temporal convolutional network with dilated convolutions
- `transformer_encoder`: self-attention encoder over channel tokens

## Next Methods to Explore

- `gru`: lighter recurrent alternative to LSTM, often faster training
- `bi_lstm`: bidirectional context over channel order, can improve reconstruction
- `graph_neural_network`: models electrode adjacency directly (strong EEG inductive bias)
- `mlp_mixer`: token-channel mixing without recurrent or convolutional blocks
- `diffusion_regressor`: iterative denoising reconstruction for robust missing-channel recovery

## Execution Convention

Each method notebook saves into its own folder:

- metrics CSV
- summary JSON
- method-specific PNG plot

This keeps experiments reproducible and avoids output file collisions.

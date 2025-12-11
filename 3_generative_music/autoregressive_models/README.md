# Auto-Regressive Models (Custom Implementation)

## Overview
Experiments with custom music generation models trained from scratch.

## Experiments

### 1. LSTM on MAESTRO (Melody-Harmony Separation)
- **Approach:** Sequence-to-sequence LSTM with melody and harmony split using music21
- **Dataset:** MAESTRO (Classical piano performances)
- **Challenge:** Excessive rest notes in separated tracks, failed to capture musicality of classical piano
- **Result:** Generated sequences with poor musical coherence
- **Status:** Abandoned - Melody separation approach unsuitable for dense classical piano

### 2. LSTM on MAESTRO (Chordify Approach)
- **Approach:** Sequence-to-sequence LSTM using chordify to process entire piece holistically
- **Dataset:** MAESTRO (Classical piano performances)
- **Result:** Generated more coherent sequences compared to melody separation
- **Challenge:** No mechanism for emotion conditioning
- **Status:** Abandoned - Lacks emotion control necessary for therapeutic application

### 3. VAE on MAESTRO
- **Approach:** Variational Autoencoder for latent emotion manipulation
- **Challenge:** Posterior collapse - model ignored latent space
- **Status:** Failed - Could not resolve collapse

## Key Learnings
Creating music generation models from scratch requires:
- Much larger datasets (20,000+ hours)
- Significant computational resources
- Extensive hyperparameter tuning

**Conclusion:** Pivoted to fine-tuning pretrained models (MusicGen)

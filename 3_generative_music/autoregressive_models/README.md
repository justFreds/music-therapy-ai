# \# Auto-Regressive Models (Custom Implementation)

# 

# \## Overview

# Experiments with custom music generation models trained from scratch.

# 

# \## Experiments

# 

# \### 1. LSTM on MAESTRO

# \- \*\*Approach:\*\* Sequence-to-sequence LSTM for piano MIDI generation

# \- \*\*Dataset:\*\* MAESTRO (Classical piano performances)

# \- \*\*Result:\*\* Generated coherent sequences but no emotion conditioning

# \- \*\*Status:\*\* ❌ Abandoned - No emotion control

# 

# \### 2. VAE on MAESTRO

# \- \*\*Approach:\*\* Variational Autoencoder for latent emotion manipulation

# \- \*\*Challenge:\*\* Posterior collapse - model ignored latent space

# \- \*\*Status:\*\* ❌ Failed - Could not resolve collapse

# 

# \## Key Learnings

# Creating music generation models from scratch requires:

# \- Much larger datasets (20,000+ hours)

# \- Significant computational resources

# \- Extensive hyperparameter tuning

# 

# \*\*Conclusion:\*\* Pivoted to fine-tuning pretrained models (MusicGen)


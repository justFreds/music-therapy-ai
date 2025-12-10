# Transformer Models - MusicGen Fine-Tuning

## Overview
Fine-tuned Meta's MusicGen on EMOTIFY dataset for emotion-conditioned music generation.

## Dataset
- **Name:** EMOTIFY
- **Size:** 400 tracks (~6.75 hours)
- **Genres:** Classical, Electronic, Pop, Rock
- **Emotions:** 9 dimensions (nostalgia, tenderness, calmness, power, etc.)

## Model
- **Base:** facebook/musicgen-small (300M parameters)
- **Approach:** Fine-tuning with emotion-conditioned text descriptions
- **Training Time:** 28 hours on RTX 3080

## Training Configuration
```python
LEARNING_RATE = 5e-6
WARMUP_STEPS = 50
BATCH_SIZE = 4
EPOCHS = 20
MAX_GRAD_NORM = 1.0
```

## Results
- **Final Loss:** 4.219
- **Success:** ✅ Different emotions produce distinct outputs
- **Emotion Vocabulary:** Model learned EMOTIFY-specific terms

## Generated Samples
Listen to comparisons in `generated_samples/`:
- `comparison_base/` - Base MusicGen outputs
- `comparison_finetuned/` - Fine-tuned model outputs

## Usage

### Jupyter Notebook
```bash
jupyter notebook musicgen_emotify/musicgen_emotify.ipynb
```

## Challenges Solved
1. ✅ NaN in logits (AudioCraft mask handling)
2. ✅ Gradient explosion (LR warmup + clipping)
3. ✅ Dropout instability (Disabled 96 dropout layers)
4. ✅ DataLoader crash (Windows multiprocessing)
5. ✅ CFG dropout causing NaN

# Music Emotion Recognition

## Overview
Comparative analysis of deep learning architectures for classifying emotions in music using the EMOTIFY dataset.

## Dataset
- **Name:** EMOTIFY
- **Size:** 400 audio tracks
- **Genres:** Classical, Electronic, Pop, Rock
- **Emotion Classes:** 9 emotions
  - Calmness (98 samples)
  - Joyful Activation (93 samples)
  - Nostalgia (53 samples)
  - Tension (44 samples)
  - Power (34 samples)
  - Solemnity (29 samples)
  - Tenderness (27 samples)
  - Sadness (19 samples)
  - Amazement (3 samples)

## Models Tested

| Model | Accuracy | Key Features |
|-------|----------|-------------|
| Hybrid LSTM-CNN | 60% | Basic audio features (MFCCs, spectral) |
| LSTM-CNN (Advanced Features) | 58% | Extended feature set |
| **Bidirectional LSTM-CNN** | 67% | Bidirectional temporal modeling |
| 2D CNN | 64% | Mel-spectrogram processing |
| 2D CNN + SpecAugmentation | 69% | Data augmentation for spectrograms |
| **Ensemble (Best)** | **76%** | BiLSTM-CNN + CNN-SpecAug combined |

## Key Findings

### Best Performing Model
**Ensemble Model (76% accuracy)** combining:
- Bidirectional LSTM-CNN hybrid
- 2D CNN with SpecAugmentation

### Architecture Insights
1. **Bidirectional processing helps** - BiLSTM-CNN outperformed unidirectional (67% vs 60%)
2. **SpecAugmentation improves robustness** - Added 5% accuracy boost (69% vs 64%)
3. **Ensemble learning effective** - Combining models gained 7% over best single model
4. **2D CNNs on spectrograms work well** - Captures both time and frequency patterns

### Feature Engineering
- **Basic features:** MFCCs, spectral centroids, chroma, zero-crossing rate
- **Advanced features:** Tonnetz, spectral contrast, harmonic/percussive separation
- Interestingly, advanced features slightly decreased accuracy (58% vs 60%), suggesting simpler features may be more robust for this dataset

### Challenges
- **Class imbalance** - Used SMOTE for balancing
- **Small dataset** - 400 samples limits deep learning potential
- **Overlapping emotions** - Some emotions are perceptually similar (e.g., calmness vs. tenderness)

## Technical Details

### Audio Processing
- **Sampling rate:** 22,050 Hz
- **Representations:**
  - Mel-spectrograms (128 bands)
  - MFCCs (13 coefficients)
  - Spectral features

### Training Configuration
- **Data split:** 80% train, 20% test
- **Data balancing:** SMOTE
- **Feature scaling:** StandardScaler
- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Validation:** Stratified K-fold

### Model Architectures

**Bidirectional LSTM-CNN (67%):**
```
Conv1D (256 filters) → BatchNorm → MaxPool
Conv1D (128 filters) → BatchNorm → MaxPool
Bidirectional LSTM (128 units)
Dense (128) → Dropout
Dense (9 classes, softmax)
```

**2D CNN with SpecAugmentation (69%):**
```
SpecAugment (time/frequency masking)
Conv2D (32 filters) → BatchNorm → MaxPool
Conv2D (64 filters) → BatchNorm → MaxPool
Conv2D (128 filters) → BatchNorm → MaxPool
Flatten → Dense (256) → Dropout
Dense (9 classes, softmax)
```

### Run the Notebook
```bash
jupyter notebook notebooks/lstm_cnn.ipynb
```

## Requirements

```bash
pip install tensorflow librosa numpy pandas scikit-learn matplotlib seaborn tqdm imbalanced-learn
```
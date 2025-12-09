# EEG-Based Emotion Classification

## Overview
Comparative analysis of machine learning models for emotion recognition from EEG signals.

## Dataset
- **Source:** Kaggle EEG Brainwave - Feeling Emotions
- **Link:** [Add Kaggle link]

## Models Tested
| Model | Accuracy | Training Time |
|-------|----------|---------------|
| SVM | 97% | < 1s |
| 1D CNN | 98% | 2.5 min |
| LSTM | 88% | 1.25 hrs |
| GRU | 90% | 1 hr |
| CNN-LSTM | 95% | 45 min |

## Key Findings
- 1D CNN achieves best accuracy-efficiency tradeoff
- SVM surprisingly competitive for this task
- Recurrent models underperform despite temporal nature

## Usage
```bash
jupyter notebook notebooks/eeg_model_comparison.ipynb
```

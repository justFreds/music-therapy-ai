# Generative AI for Music Therapy
*Systematic Review & Implementation Project*

## Project Overview
This repository contains the complete implementation and analysis for the systematic review:
**"Generative AI and Machine Learning Techniques to Enhance Music Therapy Through Personalization"**

## Project Goals
1. **EEG-based emotion recognition** - Classify emotional states from brain signals
2. **Music emotion recognition** - Understand emotional content in music
3. **Generative music systems** - Generate emotion-conditioned therapeutic music

## Repository Structure

### 1️ EEG Emotion Classification
Comparative analysis of ML models (SVM, CNN, LSTM) for EEG-based emotion recognition
- **Best Model:** 1D CNN (98% accuracy, 2.5min training)
- **Dataset:** Kaggle EEG Brainwave

### 2️ Music Emotion Recognition
Analysis of music emotion classification techniques

### 3️ Generative Music
#### 3a. Auto-Regressive Models (Custom)
- LSTM on MAESTRO → Failed to achieve emotion control
- VAE on MAESTRO → Posterior collapse issues
- **Conclusion:** Dataset too small (400 samples vs 20,000 hours needed)

#### 3b. Transformer Models (MusicGen)
- **Successfully fine-tuned MusicGen on EMOTIFY**
- Loss: 4.219
- Different emotions produce distinct musical outputs
- Learned emotion vocabulary: nostalgia, tenderness, calmness, power, etc.

## 📊 Key Results

| Component | Approach | Result |
|-----------|----------|--------|
| EEG Classification | 1D CNN | 98% accuracy |
| Custom Generation | LSTM/VAE from scratch | ❌ Failed |
| MusicGen Fine-tuning | Transfer learning | ✅ Success |

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run EEG Classification
```bash
cd 1_eeg_emotion_classification
jupyter notebook notebooks/eeg_model_comparison.ipynb
```

### Generate Emotion-Conditioned Music
```bash
cd 3_generative_music/3b_transformer_models
python musicgen_emotify/generate.py \
    --prompt "A calm classical piece with tenderness" \
    --duration 10
```

## 📄 Documentation
- [Systematic Review (PDF)](docs/)

## 🔗 Related Work
- [MusicGen Paper](https://arxiv.org/abs/2306.05284)
- [EMOTIFY Dataset](https://www.kaggle.com/datasets/yash9439/emotify-emotion-classificaiton-in-songs)
- [DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- [MAESTRO Dataset](https://magenta.withgoogle.com/datasets/maestro)

## Author
**Farid Vakili**
- Cal Poly Pomona, Computer Science
- Senior Thesis: Music Therapy Personalization
- Email: fvakili@cpp.edu f_vakili@hotmail.com

## Citation
```bibtex
@thesis{your-thesis-2024,
  author = {Your Name},
  title = {Generative AI and ML for Personalized Music Therapy},
  school = {California State Polytechnic University, Pomona},
  year = {2024}
}
```

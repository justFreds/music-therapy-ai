# \# Generative AI for Music Therapy

# \*Systematic Review \& Implementation Project\*

# 

# \## Project Overview

# This repository contains the complete implementation and analysis for the systematic review:

# \*\*"Generative AI and Machine Learning Techniques to Enhance Music Therapy Through Personalization"\*\*

# 

# \## Project Goals

# 1\. \*\*EEG-based emotion recognition\*\* - Classify emotional states from brain signals

# 2\. \*\*Music emotion recognition\*\* - Understand emotional content in music

# 3\. \*\*Generative music systems\*\* - Generate emotion-conditioned therapeutic music

# 

# \## Repository Structure

# 

# \### 1 EEG Emotion Classification

# Comparative analysis of ML models (SVM, CNN, LSTM) for EEG-based emotion recognition

# \- \*\*Best Model:\*\* 1D CNN (98% accuracy, 2.5min training)

# \- \*\*Dataset:\*\* Kaggle EEG Brainwave

# 

# \### 2 Music Emotion Recognition

# Analysis of music emotion classification techniques

# 

# \### 3 Generative Music

# \#### 3a. Auto-Regressive Models (Custom)

# \- LSTM on MAESTRO → Failed to achieve emotion control

# \- VAE on MAESTRO → Posterior collapse issues

# \- \*\*Conclusion:\*\* Dataset too small (400 samples vs 20,000 hours needed)

# 

# \#### 3b. Transformer Models (MusicGen)

# \- \*\*Successfully fine-tuned MusicGen on EMOTIFY\*\*

# \- Loss: 4.219

# \- Different emotions produce distinct musical outputs

# \- Learned emotion vocabulary: nostalgia, tenderness, calmness, power, etc.

# 

# \## Key Results

# 

# | Component | Approach | Result |

# |-----------|----------|--------|

# | EEG Classification | 1D CNN | 98% accuracy |

# | Custom Generation | LSTM/VAE from scratch | ❌ Failed |

# | MusicGen Fine-tuning | Transfer learning | ✅ Success |

# 

# \## Quick Start

# 

# \### Prerequisites

# ```bash

# pip install -r requirements.txt

# ```

# 

# \### Run EEG Classification

# ```bash

# cd 1\_eeg\_emotion\_classification

# jupyter notebook notebooks/eeg\_model\_comparison.ipynb

# ```

# 

# \### Generate Emotion-Conditioned Music

# ```bash

# cd 3\_generative\_music/3b\_transformer\_models

# python musicgen\_emotify/generate.py \\

# &nbsp;   --prompt "A calm classical piece with tenderness" \\

# &nbsp;   --duration 10

# ```

# 

# \## Documentation

# \- \[Systematic Review (PDF)](docs/)

# 

# \## Related Work

# \- \[MusicGen Paper](https://arxiv.org/abs/2306.05284)

# \- \[EMOTIFY Dataset](https://www.kaggle.com/datasets/yash9439/emotify-emotion-classificaiton-in-songs)

# \- \[DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

# \- \[MAESTRO Dataset](https://magenta.withgoogle.com/datasets/maestro)

# 

# \## Author

# \*\*Your Name\*\*

# \- Cal Poly Pomona, Computer Science

# \- Senior Thesis: Music Therapy Personalization

# \- Email: your.email@example.com

# 

# \## Citation

# ```bibtex

# @thesis{your-thesis-2024,

# &nbsp; author = {Your Name},

# &nbsp; title = {Generative AI and ML for Personalized Music Therapy},

# &nbsp; school = {California State Polytechnic University, Pomona},

# &nbsp; year = {2024}

# }

# ```

# 

# \## License

# MIT License - See LICENSE file

# 

# \## Acknowledgments

# \- Meta AI for MusicGen

# \- EMOTIFY dataset creators

# \- Cal Poly Pomona CS Department


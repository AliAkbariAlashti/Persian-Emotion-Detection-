Persian Text Emotion Classification 🎭
A comprehensive Persian text emotion classification system implementing multiple machine learning approaches for the Arman Text Emotion dataset.
🔍 Overview
This project explores three different approaches for Persian emotion classification, progressing from traditional ML to deep learning and ensemble methods:

TF-IDF + Logistic Regression - Classical approach with n-gram features
BiLSTM + Word2Vec - Deep learning with custom embeddings
Stacking Ensemble - Hybrid approach combining TF-IDF and FastText

📊 Dataset

Source: Arman Text Emotion Dataset
Classes: 7 emotions (ANGRY, FEAR, HAPPY, HATE, OTHER, SAD, SURPRISE)
Format: TSV files with text-label pairs
Language: Persian/Farsi

🚀 Methodology
Stage 1: TF-IDF + Logistic Regression

Preprocessing: Persian text normalization, stopword removal
Features: Tested unigrams, bigrams, and trigrams
Model: Logistic Regression with cross-validation
Accuracy: 40%

Stage 2: BiLSTM + Word2Vec

Embeddings: Custom Word2Vec (300d) trained on corpus
Architecture: Bidirectional LSTM with GlobalMaxPool
Challenge: Initial overfitting with 128 units + 0.3 dropout
Solution: Reduced to 64 units + increased dropout (0.5, 0.3)
Accuracy: 38.14%

Stage 3: Stacking Ensemble

Features: Combined TF-IDF (5k features) + FastText embeddings (100d)
Base Models: LogisticRegression, LinearSVC, RandomForest
Meta-learner: LogisticRegression with 5-fold CV
Accuracy: 50%

📈 Results Comparison
ModelAccuracyBest ClassesChallengesTF-IDF40%OTHER (86% recall)HAPPY (18% recall)BiLSTM38%FEAR (51% recall)HATE (8% recall)Ensemble50%FEAR (68% recall)SURPRISE (29% recall)
🔧 Key Features

Persian Text Processing: Comprehensive normalization and tokenization
Multiple Vectorization: TF-IDF, Word2Vec, FastText
Overfitting Mitigation: Regularization, dropout, early stopping
Evaluation: Detailed confusion matrices and classification reports
Prediction Functions: Ready-to-use inference capabilities

💡 Key Insights

Ensemble methods consistently outperform individual models
Persian preprocessing is crucial for performance
Class imbalance affects model performance significantly
Feature combination (TF-IDF + embeddings) provides best results

🛠️ Technologies Used

Libraries: scikit-learn, TensorFlow/Keras, Gensim
NLP: TF-IDF, Word2Vec, FastText
Models: Logistic Regression, BiLSTM, Stacking Classifier
Visualization: Matplotlib, Seaborn

📝 Usage
python# Quick prediction
result = predict_emotion("امروز روز بسیار خوبی بود و خیلی خوشحالم")
print(result['prediction'])  # Expected: HAPPY
🎯 Future Improvements

Transformer-based models (BERT, RoBERTa)
Data augmentation techniques
Advanced ensemble methods
Cross-lingual transfer learning


Achieving 50% accuracy on Persian emotion classification through systematic model progression and ensemble learning.

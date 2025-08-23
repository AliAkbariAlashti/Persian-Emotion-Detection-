
````markdown
# Persian Text Emotion Classification 🎭

A comprehensive Persian text emotion classification system implementing multiple machine learning approaches for the **ArmanEmo** dataset.

---

## 🔍 Overview

This project explores three different approaches for Persian emotion classification, progressing from traditional ML to deep learning and ensemble methods:

1. **TF-IDF + Logistic Regression** – Classical approach with n-gram features  
2. **BiLSTM + Word2Vec** – Deep learning with custom embeddings  
3. **Stacking Ensemble** – Hybrid approach combining TF-IDF and FastText  

---

## 📊 Dataset

- **Source:** [ArmanEmo: A Persian Dataset for Text-based Emotion Detection](https://github.com/Arman-Rayan-Sharif/arman-text-emotion)  
- **Authors:** Hossein Mirzaee, Javad Peymanfard, Hamid Habibzadeh Moshtaghin, Hossein Zeinali  
- **Paper:** [ArXiv:2207.11808](https://arxiv.org/abs/2207.11808)  
- **License:** Creative Commons Attribution Non Commercial Share Alike 4.0 International  
- **Classes:** 7 emotions *(ANGRY, FEAR, HAPPY, HATE, OTHER, SAD, SURPRISE)*  
- **Format:** TSV files with text-label pairs  
- **Language:** Persian/Farsi  

---

## 🚀 Methodology

### **Stage 1: TF-IDF + Logistic Regression**
- Preprocessing: Persian text normalization, stopword removal  
- Features: Tested unigrams, bigrams, trigrams  
- Model: Logistic Regression with cross-validation  
- **Accuracy:** 40%  

---

### **Stage 2: BiLSTM + Word2Vec**
- Embeddings: Custom Word2Vec (300d) trained on corpus  
- Architecture: Bidirectional LSTM with GlobalMaxPool  
- Challenge: Overfitting with 128 units + 0.3 dropout  
- Solution: Reduced to 64 units + increased dropout (0.5, 0.3)  
- **Accuracy:** 38.14%  

---

### **Stage 3: Stacking Ensemble**
- Features: Combined TF-IDF (5k features) + FastText embeddings (100d)  
- Base Models: LogisticRegression, LinearSVC, RandomForest  
- Meta-learner: LogisticRegression with 5-fold CV  
- **Accuracy:** 50%  

---

## 📈 Results Comparison

| Model   | Accuracy | Best Classes         | Challenges                |
|---------|----------|----------------------|---------------------------|
| TF-IDF  | 40%      | OTHER (86% recall)  | HAPPY (18% recall)        |
| BiLSTM  | 38%      | FEAR (51% recall)   | HATE (8% recall)          |
| Ensemble| 50%      | FEAR (68% recall)   | SURPRISE (29% recall)     |

---

## 🔧 Key Features
- Persian Text Processing: Comprehensive normalization & tokenization  
- Multiple Vectorization: TF-IDF, Word2Vec, FastText  
- Overfitting Mitigation: Regularization, dropout, early stopping  
- Evaluation: Confusion matrices & classification reports  
- Prediction Functions: Ready-to-use inference  

---

## 💡 Key Insights
- Ensemble methods consistently outperform individual models  
- Persian preprocessing is crucial for performance  
- Class imbalance significantly affects results  
- Feature combination (TF-IDF + embeddings) provides best accuracy  

---

## 🛠️ Technologies Used
- **Libraries:** scikit-learn, TensorFlow/Keras, Gensim  
- **NLP:** TF-IDF, Word2Vec, FastText  
- **Models:** Logistic Regression, BiLSTM, Stacking Classifier  
- **Visualization:** Matplotlib, Seaborn  

---

## 📝 Usage

```python
# Dataset source: https://github.com/Arman-Rayan-Sharif/arman-text-emotion
# Original authors: Mirzaee et al. (2022) - Licensed under CC BY-NC-SA 4.0

# Quick prediction
result = predict_emotion("امروز روز بسیار خوبی بود و خیلی خوشحالم")
print(result['prediction'])  # Expected: HAPPY
````

---

## 🎯 Future Improvements

* Transformer-based models (BERT, RoBERTa)
* Data augmentation techniques
* Advanced ensemble methods
* Cross-lingual transfer learning

---

## 📜 Citation

If you use this work, please cite the original **ArmanEmo** dataset:

```bibtex
@misc{mirzaee2022armanemo,
  doi = {10.48550/ARXIV.2207.11808},
  url = {https://arxiv.org/abs/2207.11808},
  author = {Mirzaee, Hossein and Peymanfard, Javad and Moshtaghin, Hamid Habibzadeh and Zeinali, Hossein},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI)},
  title = {ArmanEmo: A Persian Dataset for Text-based Emotion Detection},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}
```

---

## 🙏 Acknowledgments

We acknowledge **Hossein Mirzaee, Javad Peymanfard, Hamid Habibzadeh Moshtaghin, and Hossein Zeinali** for creating and sharing the ArmanEmo dataset.
Special thanks to the **Arman-Rayan-Sharif** organization for maintaining the dataset repository.

---

⚡ *Achieving 50% accuracy on Persian emotion classification through systematic model progression and ensemble learning, built upon the foundation of the ArmanEmo dataset.*
```

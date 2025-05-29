# 📰 Fake News Prediction System

This project is a simple yet effective machine learning model that predicts whether a news article is real or **fake**. It uses **Natural Language Processing (NLP)** techniques to preprocess text data and a **Logistic Regression** classifier for binary classification.

---

## 🚀 Features

- Text preprocessing using **NLTK** (stopword removal, stemming)
- Feature extraction using **TF-IDF Vectorization**
- Model training and evaluation using **Logistic Regression**
- Accuracy evaluation using **scikit-learn** metrics

---

## 🛠️ Libraries Used

```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

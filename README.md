# 🧠 Hybrid Random Forest + Extra Trees for Fake News Detection

## 📌 Overview

This project presents a **hybrid ensemble learning approach** combining **Random Forest (RF)** and **Extra Trees (ET)** classifiers to improve the performance of fake news detection.

The model leverages the strengths of both algorithms:

* 🌲 **Random Forest** → Reduces variance through bagging
* 🌳 **Extra Trees** → Increases randomness for better generalization

By combining them, we achieve a **more robust and accurate classification system**.



## 🚀 Key Features

* ✅ Hybrid Ensemble Model (RF + ET)
* ✅ Improved accuracy and stability
* ✅ Scalable pipeline for large datasets
* ✅ Easy-to-extend modular codebase
* ✅ Supports structured / text-based features

---

## 🏗️ Project Structure
'''
```
Hybrid-RF-ET-FakeNews/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── Hybrid_RF_ET.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_extraction.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── metrics.txt
│   └── confusion_matrix.png
│
├── requirements.txt
├── main.py
└── README.md
```

'''
## ⚙️ Methodology
### 1️⃣ Data Preprocessing

* Handle missing values
* Normalize numerical features
* Convert text using TF-IDF (if applicable)
---

### 2️⃣ Feature Extraction

* Text → TF-IDF Vectorization
* Numerical → Standard Scaling
* Combine features into a unified representation

---

### 3️⃣ Model Training

#### 🌲 Random Forest

* Ensemble of decision trees using bagging
* Reduces overfitting

#### 🌳 Extra Trees

* Randomized splits for higher diversity
* Faster and more generalized

---
### 4️⃣ Hybrid Ensemble (Core Contribution)

We combine both models using **soft voting**:

---



python
from sklearn.ensemble import VotingClassifier

hybrid_model = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier()),
        ("et", ExtraTreesClassifier())
    ],
    voting="soft"
)
---

---

### 5️⃣ Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

## 📊 Results

| Model          | Performance |
| -------------- | ----------- |
| Random Forest  | Baseline    |
| Extra Trees    | Baseline    |
| ✅ Hybrid Model | **Best**    |

> The hybrid model consistently outperforms individual classifiers in terms of accuracy and generalization.

---

## 🧪 How to Run

### 🔹 1. Clone Repository

---
bash
git clone https://github.com/your-username/hybrid-rf-et-fakenews.git
cd hybrid-rf-et-fakenews
---

### 🔹 2. Install Dependencies

---bash
pip inst
all -r requirements.txt
---

### 🔹 3. Run the Project

---
bash
python main.py
---


## 📦 Requirements


 numpy
pandas
scikit-learn
matplotlib
seaborn
---

---

## 💾 Model Saving

---
python
import joblib
joblib.dump(hybrid_model, "hybrid_model.pkl")
---

---

## 📈 Future Improvements

* 🔥 Deep Learning Integration (CNN / Transformer)
* 🔥 Multimodal Learning (Text + Image)
* 🔥 Explainability (SHAP, LIME)
* 🔥 Hyperparameter Optimization
* 🔥 Deployment via Streamlit / Flask

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Md Irfanul Kabir**
AI & Machine Learning Enthusiast 🚀

---

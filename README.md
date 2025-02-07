# 🎵 Music Genre Classification

This section presents our attempt to create a baseline model for music genre classification.  
We implement machine learning models to classify music genres based on audio features.  
The models can distinguish between **10 different genres**:  
**Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, and Rock**.

---

## 🚀 Models & Performance
### 30 seconds audios results
| Model         | Macro F1 | Accuracy | Best Performing Genres (F1)     | Challenging Genres (F1) |
|--------------|----------|----------|---------------------------------|--------------------------|
| **Random Forest** | **0.75** | **0.76** | Metal (0.88), Pop (0.88), Reggae (0.81) | Country (0.56), Rock (0.61) |
| **XGBoost**  | **0.73** | **0.73** | Metal (0.90), Pop (0.90), Classical (0.77) | Rock (0.65), Country (0.56) |
| **SVM (RBF)** | **0.57** | **0.57** | Classical (0.85), Metal (0.82) | Disco (0.38), Rock (0.39) |
| **Stacking** | **0.61** | **0.62** | Metal (0.82), Classical (0.80) | Rock (0.28), Hip-hop (0.43) |

**Key Findings**:  
🌚 **Random Forest** achieves the highest **F1-score (0.75)** and most consistent results.  
🌚 **XGBoost** performs similarly but has higher variance across genres.  
⚠️ **Rock & Country** are the hardest to classify.  
🎸 **Metal & Pop** are the most reliably classified across models.

---

## 🧠 Approach  

We experimented with two feature selection approaches:  
1️⃣ **Full Feature Set**: Using all available features.  
2️⃣ **Top 8 Features**: Selected based on **F-scores** for improved efficiency.

![Top features](projet/experimentation/assets/image.png)

🔹 **Note:** While dimensionality reduction wasn't necessary due to the limited feature set,  
we included this step to highlight its importance in real-world scenarios.

---

## 🗂 Dataset  

We use the **same dataset** as the one used for our **Deep Learning models**.

---

## 📊 Performance Metrics  

To evaluate our models, we use:  
✔ **Macro-averaged F1-score**  
✔ **Per-class precision & recall**  
✔ **Confusion matrix analysis**  

---

## 📦 Requirements  

Ensure you have the following dependencies installed:  

```bash
pip install numpy pandas scikit-learn xgboost seaborn matplotlib
```

---

## 📝 Files  

| File | Description |
|------|------------|
| **models.ipynb** | Model training & evaluation |
| **EDA_num.ipynb** | Exploratory Data Analysis (numerical features) |
| **music_features_scaled.csv** | Scaled dataset with all features |
| **top_features_scaled.csv** | Dataset with selected important features |

🔹 **Files that are not critical for final results**:  
- **EDA_specs.ipynb**: Spectrogram analysis.  
- **EDA.ipynb**: Initial exploratory data analysis.

---

## 🔮 Future Improvements  

📌 **Data Augmentation** to improve model generalization.  
📌 **Deep Learning models** for enhanced feature extraction.  
📌 **Hyperparameter Optimization** for better tuning.  

---


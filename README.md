# 🧠 Predicting Match Outcomes in Professional Tennis

> **Teamproject FSS 2026** · Data and Web Science Group · Universität Mannheim

## 👥 Team 10:
Gerrit Christopher Regelmann
Eljes Basha
Lucas Waage 
Daniel Chernychenko
Tim Stelzner
Julia Kocharina

---

## 📌 Project Goal

> The objective of this project is to develop a predictive model that estimates the outcome of professional men’s ATP tennis matches. Specifically, the task is to predict the winner of a given match based on historical and contextual data available prior to the match.

---

## 🗂️ Repository Structure

project/
├── /data
│   ├── visualisations                      # visualisations (output of 01_eda_analysis.py)
│   ├── cleaned_2000_2026.csv               # preprocessed dataset
│   ├── raw_2000-2026.csv                   # raw dataset
│   ├── test.csv                            # test data (output of 03_train_test_split.py)
│   ├── train.csv                           # train data (output of 03_train_test_split.py) 
├── pipeline/
│   ├── 01_eda_analysis.py/                 # data visualisation
│   |── 02_preprocessing_feature_eng.py/    # preprocessing and feature engineering --> cleaned_2000_2026.csv
│   ├── 03_train_test_split.py              # splits dataset into train and test data
|   ├── 04_baseline.py                      # 2 baseline models (Rank Baseline, Elo Baseline)
│   |── 05_some_algoritm.py/                # tba
│   ├── 06_some_algorithm.py                # tba
|   ├── 07_some_algorithm.py                # tba
│   
├── Project_Report.pdf                      
├── Project_Presentation.pdf
├── README.md                        # project overwiew
└── requirements.txt                 # necessary packages and libraries 

```
TBA
```

---

## 📦 Dataset

This project uses **[ATP Tennis 2000 - 2026]([(https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull)])** — a large, freely available database 
| Property | Detail |
|---|---|
| **Dataset 1** | ATP Tennis 2000 - 2026 |
| **Size** | 60k+ ATP macthes(2000-2026) |
| **Dataset 2** |   |
| **Size** |  |
| **Modalities** | Time series, text data, numerical data|
| **Access** | publicly available |

### Prediction Task

| Task | Type | Label |
|---|---|---|
| **Match outcome** | Binary classification: win/loose | 0, 1 |


---

## 🛠️ Local Environment Setup

### 1️⃣ Clone the repository

```bash
git clone [git@github.com:Mummbach/teamproject_heinzl_FSS26.git]
cd DataMiningTennis
```

### 2️⃣ Create a virtual environment

```bash
python3 -m venv venv
```

### 3️⃣ Activate the virtual environment

```bash
source venv/bin/activate
```

Your prompt should change to:

```
(venv) user:~/DataMiningTennis$
```

To deactivate at any time:

```bash
deactivate
```

### 4️⃣ Install all required dependencies

```bash
pip install -r requirements.txt
```

### 🧠 Notes

- If you install new packages, update the team's dependency list:

```bash
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push
```

---






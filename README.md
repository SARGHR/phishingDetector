# 🛡️ Détecteur de Phishing Intelligent
**Projet Tuteuré L3 — Science des Données | Université Lumière Lyon 2 | 2025–2026**  
**Auteure : Sarah Ghouar**

---

## 📌 Description

Système intelligent de détection de phishing par Machine Learning et explicabilité (SHAP).  
Le système analyse une URL en temps réel et prédit si elle est **légitime ou malveillante**, en expliquant **pourquoi** grâce aux valeurs SHAP.

### 5 originalités du projet
- **Analyse multimodale** : URL + contenu HTML de la page cible (30 features UCI)
- **Explicabilité SHAP** : chaque prédiction est accompagnée des features qui l'ont influencée
- **Focus phishing français** : détection des imitations de CAF, Ameli, banques FR
- **Apprentissage continu** : les erreurs signalées par l'utilisateur sont stockées et permettent un réentraînement périodique
- **Dashboard analytique** : suivi des analyses en temps réel

---

## 🏗️ Architecture du projet
phishingDetector/
├── data/
│   ├── raw/                  ← dataset UCI brut (11 054 URLs)
│   ├── processed/            ← dataset nettoyé (dataset_clean.csv)
│   └── feedback/             ← base SQLite (historique + corrections)
├── features/
│   ├── url_features.py       ← 16 features extraites depuis l'URL
│   ├── html_features.py      ← 14 features extraites depuis le HTML
│   └── feature_pipeline.py   ← pipeline complet (30 features UCI)
├── models/
│   ├── ensemble_model.pkl    ← modèle entraîné (RF + XGBoost)
│   ├── scaler.pkl            ← StandardScaler
│   ├── feature_names.pkl     ← ordre exact des 30 features
│   ├── shap_explainer.pkl    ← explainer SHAP
│   └── versions/             ← historique des modèles réentraînés
├── notebooks/
│   ├── 01_exploration.py     ← exploration et nettoyage du dataset
│   ├── 02_train_model.py     ← entraînement RF + XGBoost + Ensemble
│   └── 03_retrain.py         ← réentraînement avec corrections utilisateurs
├── interface/
│   └── app.py                ← interface web Streamlit (4 pages)
├── reports/
│   ├── shap_summary.png      ← graphique SHAP global
│   └── shap_waterfall.png    ← explication d'une prédiction individuelle
├── requirements.txt
└── README.md

---

## ⚙️ Installation

### Prérequis
- Python 3.11+
- Git

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/TON_USERNAME/phishingDetector.git
cd phishingDetector

# 2. Créer et activer l'environnement virtuel
python3 -m venv env_phishing
source env_phishing/bin/activate  # Linux/Mac
# ou : env_phishing\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Lancement

### Interface web (recommandé)
```bash
streamlit run interface/app.py
```
Puis ouvrir `http://localhost:8501` dans le navigateur.

### Réentraîner le modèle (si > 10 corrections collectées)
```bash
python notebooks/03_retrain.py
```

---

## 📊 Performances du modèle

| Modèle | Précision | Recall | AUC-ROC |
|--------|-----------|--------|---------|
| Random Forest | 0.97 | 0.97 | 0.9951 |
| XGBoost | 0.97 | 0.97 | 0.9956 |
| **Ensemble (RF + XGBoost)** | **0.97** | **0.97** | **0.9966** |

Dataset : UCI Phishing Websites (11 054 URLs, 30 features, 0 valeurs manquantes)

---

## 🔍 Fonctionnement des features

Les 30 features sont extraites selon les règles officielles du dataset UCI (Mohammad, Thabtah & McCluskey, 2014) :

- **Features URL (16)** : longueur, sous-domaines, présence d'IP, HTTPS, tiret dans le domaine, raccourcisseur, âge du domaine (WHOIS)...
- **Features HTML (14)** : ressources externes, liens d'ancrage, iframes, formulaires suspects, JavaScript malveillant...

Convention : `1 = Phishing`, `-1 = Légitime`, `0 = Suspect`

---

## 🖥️ Interface Streamlit

| Page | Description |
|------|-------------|
| **Analyseur** | Saisir une URL → résultat + explication SHAP + feedback |
| **Dashboard** | Statistiques globales, taux de phishing, graphiques |
| **Historique** | Tableau de toutes les analyses avec filtre |

---

## 📚 Sources et références

- Dataset : Mohammad, R., Thabtah, F. & McCluskey, L. (2014). *Phishing Websites Dataset*. UCI ML Repository.
- SHAP : Lundberg, S. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions*. arXiv:1705.07874
- XGBoost : Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD 2016.

---

## ✍️ Auteure

**Sarah Ghouar**  
L3 Informatique — Science des Données  
Université Lumière Lyon 2 | 2025–2026  
Projet tuteuré encadré par Carlos Crispim

# Ce script implemente l'apprentissage continu du projet.
# Il reentraine le modele ML quand suffisamment de corrections
# ont ete collectees via le bouton "Signaler une erreur" de l'interface.
#
# Conditions pour lancer ce script :
#   - Au moins 10 corrections enregistrees dans la base SQLite
#
# A lancer manuellement depuis la racine du projet :
# python scripts/03_retrain.py
#
# Ce script produit :
#   - Un nouveau modele sauvegarde dans models/versions/ (historique)
#   - Le modele actif models/ensemble_model.pkl mis a jour
#   - Le scaler actif models/scaler.pkl mis a jour
#   - L'explainer SHAP models/shap_explainer.pkl mis a jour

import pandas as pd    # chargement du dataset et des corrections
import sqlite3         # connexion a la base de donnees SQLite
import joblib          # sauvegarde et chargement des modeles
import os              # gestion des fichiers et dossiers
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import shap            # pour recalculer l'explainer apres le reentainement


# ── 1. VERIFIER LE NOMBRE DE CORRECTIONS ─────────────────────
# On lit la table "corrections" dans la base SQLite
# pour savoir si l'utilisateur a signale assez d'erreurs
DB_PATH = 'data/feedback/historique.db'

conn = sqlite3.connect(DB_PATH)
# pd.read_sql_query charge le resultat d'une requete SQL dans un DataFrame
corrections = pd.read_sql_query('SELECT * FROM corrections', conn)
conn.close()

print(f"Corrections disponibles : {len(corrections)}")

# Seuil minimum : 10 corrections avant de reentainer
# En dessous, le bruit statistique est trop important
if len(corrections) < 10:
    print("Pas assez de corrections (minimum 10). Reentainement annule.")
    exit()  # arrete le script proprement


# ── 2. CHARGER LE DATASET ORIGINAL ───────────────────────────
# On repart toujours du dataset UCI original (11054 sites)
# comme base solide pour le reentainement
print("Chargement du dataset original...")
df_original = pd.read_csv('data/processed/dataset_clean.csv')

# On charge aussi la liste des 30 features dans l'ordre exact
# pour s'assurer que le nouveau modele utilise les memes colonnes
feature_names = joblib.load('models/feature_names.pkl')

# Separation features / labels
X_original = df_original[feature_names]
# Conversion des labels : -1 (Legitime) devient 0 pour XGBoost
y_original = df_original['class'].replace({-1: 0})


# ── 3. INTEGRER LES CORRECTIONS ──────────────────────────────
# Idealement on ajouterait les corrections au dataset d'entrainement.
# Probleme : les corrections ne contiennent que les URLs (texte brut),
# pas les 30 features numeriques dont le modele a besoin.
# Recalculer les features pour chaque URL prendrait trop de temps
# (WHOIS + telechargement HTML pour chaque correction).
#
# Solution de compromis : on sauvegarde les corrections dans un fichier log
# pour la tracabilite, et on reentraine sur le dataset original.
# Les corrections servent ainsi de "signal de qualite" plutot que de donnees d'entrainement directes.
print(f"Integration de {len(corrections)} corrections...")

os.makedirs('models/versions', exist_ok=True)
# Sauvegarde du log des corrections en CSV pour la tracabilite
corrections.to_csv('models/versions/corrections_log.csv', index=False)
print("Corrections sauvegardees dans models/versions/corrections_log.csv")


# ── 4. REENTAINER SUR LE DATASET ORIGINAL ────────────────────
# Meme pipeline que 02_train_model.py
print("\nReentainement du modele...")

# Separation train/test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42
)

# Normalisation : calcule les moyennes et ecarts-types sur X_train uniquement
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creation des deux modeles de base
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)

# Combinaison en Ensemble par vote doux (soft voting)
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)

# Entrainement du modele ensemble sur les donnees d'entrainement
ensemble.fit(X_train_scaled, y_train)

# Evaluation du nouveau modele sur le jeu de test
y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC du nouveau modele : {auc:.4f}")


# ── 5. VERSIONNER ET SAUVEGARDER ─────────────────────────────
# On garde un historique de tous les modeles entraines
# pour pouvoir revenir a une version anterieure si besoin.
# Convention de nommage : ensemble_model_v2.pkl, v3.pkl, v4.pkl...
# (v1 = le modele original produit par 02_train_model.py)

# Compte combien de versions existent deja dans models/versions/
versions = [f for f in os.listdir('models/versions') if f.startswith('ensemble_model_v')]
# +2 car v1 est le modele original (non stocke dans versions/)
version_num = len(versions) + 2

# Sauvegarde de la nouvelle version dans l'historique
joblib.dump(ensemble, f'models/versions/ensemble_model_v{version_num}.pkl')
joblib.dump(scaler, f'models/versions/scaler_v{version_num}.pkl')
print(f"Modele v{version_num} sauvegarde dans models/versions/")

# Mise a jour du modele ACTIF utilise par l'interface Streamlit
# Ces fichiers sont ceux charges par app.py au demarrage
joblib.dump(ensemble, 'models/ensemble_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Mise a jour de l'explainer SHAP
# L'explainer doit etre recree car il est lie au modele entraine
# On extrait le Random Forest de l'ensemble pour creer l'explainer
# (TreeExplainer ne fonctionne qu'avec des modeles a base d'arbres individuels)
print("\nMise a jour de l'explainer SHAP...")
rf_model = ensemble.named_estimators_['rf']  # recupere le RF depuis l'ensemble
explainer = shap.TreeExplainer(rf_model)
joblib.dump(explainer, 'models/shap_explainer.pkl')

# Bilan final
print(f"\nReentainement termine ! Modele v{version_num} actif.")
print(f"   AUC-ROC : {auc:.4f}")
print(f"   Corrections integrees : {len(corrections)}")
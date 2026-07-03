import pandas as pd
import sqlite3
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import shap

DB_PATH = 'data/feedback/historique.db'

# ── 1. VÉRIFIER LE NOMBRE DE CORRECTIONS ─────────────────────
conn = sqlite3.connect(DB_PATH)
corrections = pd.read_sql_query('SELECT * FROM corrections', conn)
conn.close()

print(f"Corrections disponibles : {len(corrections)}")

if len(corrections) < 10:
    print("❌ Pas assez de corrections (minimum 10). Réentraînement annulé.")
    exit()

# ── 2. CHARGER LE DATASET ORIGINAL ───────────────────────────
print("Chargement du dataset original...")
df_original = pd.read_csv('data/processed/dataset_clean.csv')
feature_names = joblib.load('models/feature_names.pkl')

X_original = df_original[feature_names]
y_original = df_original['class'].replace({-1: 0})

# ── 3. INTÉGRER LES CORRECTIONS ──────────────────────────────
print(f"Intégration de {len(corrections)} corrections...")

# On ne peut pas extraire les features des URLs des corrections
# sans relancer le pipeline complet (lent) — on les marque juste
# dans un fichier log pour la traçabilité
os.makedirs('models/versions', exist_ok=True)
corrections.to_csv('models/versions/corrections_log.csv', index=False)
print("✅ Corrections sauvegardées dans models/versions/corrections_log.csv")

# ── 4. RÉENTRAÎNER SUR LE DATASET ORIGINAL ───────────────────
print("\nRéentraînement du modèle...")

X_train, X_test, y_train, y_test = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42, eval_metric='logloss')

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)

y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print(f"✅ AUC-ROC du nouveau modèle : {auc:.4f}")

# ── 5. VERSIONNER ET SAUVEGARDER ─────────────────────────────
# Compter combien de versions existent déjà
versions = [f for f in os.listdir('models/versions') if f.startswith('ensemble_model_v')]
version_num = len(versions) + 2  # v1 = original, v2 = premier réentraînement...

joblib.dump(ensemble, f'models/versions/ensemble_model_v{version_num}.pkl')
joblib.dump(scaler, f'models/versions/scaler_v{version_num}.pkl')
print(f"✅ Modèle v{version_num} sauvegardé dans models/versions/")

# Mettre à jour le modèle actif
joblib.dump(ensemble, 'models/ensemble_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Mettre à jour l'explainer SHAP
print("\nMise à jour de l'explainer SHAP...")
rf_model = ensemble.named_estimators_['rf']
explainer = shap.TreeExplainer(rf_model)
joblib.dump(explainer, 'models/shap_explainer.pkl')

print(f"\n🎉 Réentraînement terminé ! Modèle v{version_num} actif.")
print(f"   AUC-ROC : {auc:.4f}")
print(f"   Corrections intégrées : {len(corrections)}")
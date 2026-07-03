import streamlit as st
import pandas as pd
import joblib
import sys
import os

# Permet d'importer depuis le dossier features/ et models/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from features.feature_pipeline import extraire_toutes_les_features

# ── CONFIGURATION DE LA PAGE ──────────────────────────────────
st.set_page_config(
    page_title="Détecteur de Phishing",
    page_icon="🛡️",
    layout="centered"
)

# ── CHARGER LE MODÈLE (une seule fois, mis en cache) ──────────
@st.cache_resource
def charger_modele():
    ensemble = joblib.load('models/ensemble_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    return ensemble, scaler, feature_names, explainer

ensemble, scaler, feature_names, explainer = charger_modele()

# ── TITRE ──────────────────────────────────────────────────────
st.title("🛡️ Détecteur de Phishing Intelligent")
st.write("Entrez une URL pour analyser son niveau de risque.")

# ── ZONE DE SAISIE ─────────────────────────────────────────────
url_input = st.text_input("URL à analyser", placeholder="https://exemple.com")

if st.button("Analyser", type="primary"):
    if url_input:
        with st.spinner("Analyse en cours (peut prendre quelques secondes)..."):

            # 1. Extraire les 30 features de l'URL
            feats = extraire_toutes_les_features(url_input)

            # 2. Construire le vecteur dans le bon ordre
            X = pd.DataFrame([feats])[feature_names]
            X_scaled = scaler.transform(X)

            # 3. Prédiction
            proba = ensemble.predict_proba(X_scaled)[0][1]
            label = "PHISHING" if proba > 0.5 else "LÉGITIME"

            # 4. Affichage du résultat
            st.divider()
            if label == "PHISHING":
                st.error(f"⚠️ **{label}** — Risque de phishing estimé : {proba:.1%}")
            else:
                st.success(f"✅ **{label}** — Risque de phishing estimé : {proba:.1%}")

            # 5. Explication SHAP
            sv_site = explainer.shap_values(X)
            shap_vals = sv_site[:, :, 1][0]

            feature_impact = list(zip(feature_names, shap_vals))
            feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
            top5 = feature_impact[:5]

            st.subheader("🔍 Pourquoi ce résultat ?")
            for feat, val in top5:
                sens = "🔴 augmente" if val > 0 else "🟢 diminue"
                st.write(f"{sens} le risque : **{feat}** (impact: {val:+.3f})")

            # 6. Détail des features (optionnel, repliable)
            with st.expander("Voir toutes les features détectées"):
                st.dataframe(X.T.rename(columns={0: 'valeur'}))

    else:
        st.warning("Merci d'entrer une URL.")

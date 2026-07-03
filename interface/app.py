import streamlit as st
import pandas as pd
import joblib
import sys
import os
import sqlite3
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from features.feature_pipeline import extraire_toutes_les_features

# ── CONFIGURATION DE LA PAGE ──────────────────────────────────
st.set_page_config(
    page_title="Détecteur de Phishing",
    page_icon="🛡️",
    layout="centered"
)

# ── BASE DE DONNÉES SQLite ────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'feedback', 'historique.db')

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            label TEXT,
            proba REAL,
            date TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            label_predit TEXT,
            label_correct TEXT,
            date TEXT
        )
    ''')
    conn.commit()
    conn.close()

def sauvegarder_analyse(url, label, proba):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO analyses (url, label, proba, date) VALUES (?, ?, ?, ?)',
        (url, label, proba, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()

def charger_historique():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM analyses ORDER BY id DESC', conn)
    conn.close()
    return df

def sauvegarder_correction(url, label_predit, label_correct):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO corrections (url, label_predit, label_correct, date) VALUES (?, ?, ?, ?)',
        (url, label_predit, label_correct, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()

def compter_corrections():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute('SELECT COUNT(*) FROM corrections')
    count = cursor.fetchone()[0]
    conn.close()
    return count

init_db()

# ── CHARGER LE MODÈLE (une seule fois, mis en cache) ──────────
@st.cache_resource
def charger_modele():
    ensemble = joblib.load('models/ensemble_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    return ensemble, scaler, feature_names, explainer

ensemble, scaler, feature_names, explainer = charger_modele()

# ── MENU LATÉRAL ───────────────────────────────────────────────
st.sidebar.title("🛡️ Navigation")
page = st.sidebar.radio("Aller à :", ["Analyseur", "Dashboard", "Historique"])


# ══════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSEUR
# ══════════════════════════════════════════════════════════════
if page == "Analyseur":
    st.title("🛡️ Détecteur de Phishing Intelligent")
    st.write("Entrez une URL pour analyser son niveau de risque.")

    url_input = st.text_input("URL à analyser", placeholder="https://exemple.com")

    if st.button("Analyser", type="primary"):
        if url_input:
            with st.spinner("Analyse en cours (peut prendre quelques secondes)..."):

                feats = extraire_toutes_les_features(url_input)
                X = pd.DataFrame([feats])[feature_names]
                X_scaled = scaler.transform(X)

                proba = ensemble.predict_proba(X_scaled)[0][1]
                label = "PHISHING" if proba > 0.5 else "LÉGITIME"

                sauvegarder_analyse(url_input, label, proba)

                st.divider()
                if label == "PHISHING":
                    st.error(f"⚠️ **{label}** — Risque de phishing estimé : {proba:.1%}")
                else:
                    st.success(f"✅ **{label}** — Risque de phishing estimé : {proba:.1%}")

                sv_site = explainer.shap_values(X)
                shap_vals = sv_site[:, :, 1][0]

                feature_impact = list(zip(feature_names, shap_vals))
                feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
                top5 = feature_impact[:5]

                st.subheader("🔍 Pourquoi ce résultat ?")
                for feat, val in top5:
                    sens = "🔴 augmente" if val > 0 else "🟢 diminue"
                    st.write(f"{sens} le risque : **{feat}** (impact: {val:+.3f})")

                with st.expander("Voir toutes les features détectées"):
                    st.dataframe(X.T.rename(columns={0: 'valeur'}))
                # 7. Bouton feedback — apprentissage continu
                st.divider()
                st.subheader("🔁 Cette prédiction est-elle correcte ?")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Oui, c'est correct"):
                        st.success("Merci pour votre retour !")

                with col2:
                    if st.button("❌ Non, signaler une erreur"):
                        label_correct = "LÉGITIME" if label == "PHISHING" else "PHISHING"
                        sauvegarder_correction(url_input, label, label_correct)
                        nb = compter_corrections()
                        st.warning(f"Correction enregistrée ! ({nb} corrections au total)")
                        if nb >= 10:
                            st.info("💡 Seuil de 10 corrections atteint — vous pouvez lancer le réentraînement depuis le terminal avec : `python notebooks/03_retrain.py`")
        else:
            st.warning("Merci d'entrer une URL.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.title("📊 Dashboard Analytique")

    df_hist = charger_historique()

    if df_hist.empty:
        st.info("Aucune analyse enregistrée pour le moment. Va sur la page Analyseur pour commencer !")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total analyses", len(df_hist))
        with col2:
            nb_phishing = (df_hist['label'] == 'PHISHING').sum()
            st.metric("Phishing détectés", nb_phishing)
        with col3:
            taux = nb_phishing / len(df_hist) * 100
            st.metric("Taux de phishing", f"{taux:.1f}%")

        st.divider()

        st.subheader("Répartition des résultats")
        repartition = df_hist['label'].value_counts()
        st.bar_chart(repartition)

        st.subheader("Évolution du risque moyen dans le temps")
        df_hist_sorted = df_hist.sort_values('id')
        st.line_chart(df_hist_sorted.set_index('id')['proba'])


# ══════════════════════════════════════════════════════════════
# PAGE 3 — HISTORIQUE
# ══════════════════════════════════════════════════════════════
elif page == "Historique":
    st.title("📜 Historique des analyses")

    df_hist = charger_historique()

    if df_hist.empty:
        st.info("Aucune analyse enregistrée pour le moment.")
    else:
        filtre = st.selectbox("Filtrer par résultat :", ["Tous", "PHISHING", "LÉGITIME"])

        if filtre != "Tous":
            df_affiche = df_hist[df_hist['label'] == filtre]
        else:
            df_affiche = df_hist

        df_affiche_display = df_affiche.copy()
        df_affiche_display['proba'] = df_affiche_display['proba'].apply(lambda x: f"{x:.1%}")
        df_affiche_display = df_affiche_display.rename(columns={
            'url': 'URL',
            'label': 'Résultat',
            'proba': 'Confiance',
            'date': 'Date'
        })

        st.dataframe(
            df_affiche_display[['URL', 'Résultat', 'Confiance', 'Date']],
            use_container_width=True
        )
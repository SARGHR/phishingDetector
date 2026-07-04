# Ce fichier est l'interface web du projet.
# Il est lance avec la commande : streamlit run interface/app.py
# Streamlit transforme ce script Python en une application web interactive
# accessible depuis le navigateur a l'adresse http://localhost:8501
#
# L'application contient 3 pages :
#   - Analyseur  : saisir une URL et obtenir une prediction + explication SHAP
#   - Dashboard  : statistiques globales sur toutes les analyses effectuees
#   - Historique : tableau de toutes les analyses avec filtre par resultat
#
# Les analyses et les corrections sont stockees dans une base SQLite locale.

import streamlit as st  # framework pour creer des interfaces web en Python
import pandas as pd     # manipulation des donnees (DataFrames)
import joblib           # chargement des modeles ML sauvegardes (.pkl)
import sys              # pour modifier le chemin de recherche des modules Python
import os               # pour gerer les chemins de fichiers
import sqlite3          # base de donnees locale (pas besoin de serveur)
from datetime import datetime  # pour horodater les analyses et corrections

# Ajoute le dossier parent au chemin de recherche Python
# pour pouvoir importer feature_pipeline depuis n'importe ou
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from features.feature_pipeline import extraire_toutes_les_features


# ── CONFIGURATION DE LA PAGE ──────────────────────────────────
# Cette fonction DOIT etre appelee en premier avant tout autre appel Streamlit
st.set_page_config(
    page_title="Detecteur de Phishing",  # titre affiché dans l'onglet du navigateur
    page_icon="shield",                  # icone de l'onglet
    layout="centered"                    # contenu centre dans la page
)


# ── BASE DE DONNEES SQLite ────────────────────────────────────
# Chemin vers le fichier de base de donnees SQLite
# os.path.join construit le chemin de facon compatible Linux/Windows
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'feedback', 'historique.db')


def init_db():
    # Cree la base de donnees et les tables si elles n'existent pas encore.
    # Cette fonction est appelee au demarrage de l'application.
    # "CREATE TABLE IF NOT EXISTS" evite une erreur si la table existe deja.
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # Table analyses : stocke chaque URL analysee avec son resultat
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,        -- l'URL analysee
            label TEXT,      -- "PHISHING" ou "LEGITIME"
            proba REAL,      -- probabilite de phishing entre 0 et 1
            date TEXT        -- horodatage de l'analyse
        )
    ''')

    # Table corrections : stocke les erreurs signalees par l'utilisateur
    # (module d'apprentissage continu)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,            -- l'URL mal classee
            label_predit TEXT,   -- ce que le modele avait predit
            label_correct TEXT,  -- la vraie reponse selon l'utilisateur
            date TEXT            -- horodatage de la correction
        )
    ''')
    conn.commit()  # valide les modifications
    conn.close()   # ferme la connexion


def sauvegarder_analyse(url, label, proba):
    # Insere une nouvelle ligne dans la table "analyses".
    # Appelee apres chaque analyse sur la page Analyseur.
    # Les ? sont des parametres lies -> protection contre les injections SQL
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO analyses (url, label, proba, date) VALUES (?, ?, ?, ?)',
        (url, label, proba, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()


def charger_historique():
    # Charge toutes les analyses depuis la base SQLite
    # et les retourne sous forme de DataFrame pandas.
    # ORDER BY id DESC = les plus recentes en premier
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM analyses ORDER BY id DESC', conn)
    conn.close()
    return df


def sauvegarder_correction(url, label_predit, label_correct):
    # Insere une correction dans la table "corrections".
    # Appelee quand l'utilisateur clique sur "Non, signaler une erreur".
    # Ces corrections servent au reentainement periodique du modele (03_retrain.py).
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        'INSERT INTO corrections (url, label_predit, label_correct, date) VALUES (?, ?, ?, ?)',
        (url, label_predit, label_correct, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    )
    conn.commit()
    conn.close()


def compter_corrections():
    # Retourne le nombre total de corrections enregistrees dans la base.
    # Utilise pour afficher un message quand le seuil de 10 corrections est atteint.
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute('SELECT COUNT(*) FROM corrections')
    count = cursor.fetchone()[0]  # fetchone() recupere la premiere (et unique) ligne
    conn.close()
    return count


# Initialise la base de donnees au demarrage (cree les tables si necessaire)
init_db()


# ── CHARGER LE MODELE ─────────────────────────────────────────
# @st.cache_resource : charge le modele UNE SEULE FOIS au demarrage
# et le garde en memoire pour toutes les analyses suivantes.
# Sans ce cache, le modele serait recharge a chaque clic -> tres lent.
@st.cache_resource
def charger_modele():
    ensemble = joblib.load('models/ensemble_model.pkl')      # modele RF + XGBoost entraine
    scaler = joblib.load('models/scaler.pkl')                # StandardScaler entraine
    feature_names = joblib.load('models/feature_names.pkl')  # ordre exact des 30 features
    explainer = joblib.load('models/shap_explainer.pkl')     # explainer SHAP
    return ensemble, scaler, feature_names, explainer

# Chargement au demarrage (une seule fois grace au cache)
ensemble, scaler, feature_names, explainer = charger_modele()


# ── MENU LATERAL ──────────────────────────────────────────────
# st.sidebar = panneau gauche de l'interface
# st.sidebar.radio = boutons radio pour naviguer entre les pages
# La valeur selectionnee est stockee dans la variable "page"
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller a :", ["Analyseur", "Dashboard", "Historique"])


# ══════════════════════════════════════════════════════════════
# PAGE 1 — ANALYSEUR
# ══════════════════════════════════════════════════════════════
if page == "Analyseur":
    st.title("Detecteur de Phishing Intelligent")
    st.write("Entrez une URL pour analyser son niveau de risque.")

    # Champ de saisie pour l'URL
    url_input = st.text_input("URL a analyser", placeholder="https://exemple.com")

    # Bouton principal d'analyse
    if st.button("Analyser", type="primary"):
        if url_input:
            # st.spinner affiche un message de chargement pendant le calcul
            with st.spinner("Analyse en cours (peut prendre quelques secondes)..."):

                # ETAPE 1 : extraire les 30 features de l'URL
                feats = extraire_toutes_les_features(url_input)

                # ETAPE 2 : construire le DataFrame dans le bon ordre de colonnes
                X = pd.DataFrame([feats])[feature_names]

                # ETAPE 3 : normaliser les features (meme echelle que lors de l'entrainement)
                X_scaled = scaler.transform(X)

                # ETAPE 4 : predire la probabilite de phishing
                # predict_proba retourne [[proba_legitime, proba_phishing]]
                # [0][1] recupere uniquement la proba de phishing
                proba = ensemble.predict_proba(X_scaled)[0][1]
                label = "PHISHING" if proba > 0.5 else "LEGITIME"

                # ETAPE 5 : sauvegarder l'analyse dans la base SQLite
                sauvegarder_analyse(url_input, label, proba)

                # ETAPE 6 : afficher le resultat
                # st.divider() trace une ligne horizontale de separation
                st.divider()
                if label == "PHISHING":
                    # st.error = encadre rouge
                    st.error(f"ATTENTION - {label} - Risque de phishing estime : {proba:.1%}")
                else:
                    # st.success = encadre vert
                    st.success(f"OK - {label} - Risque de phishing estime : {proba:.1%}")

                # ETAPE 7 : calcul et affichage de l'explication SHAP
                # shap_values retourne les contributions de chaque feature a la prediction
                sv_site = explainer.shap_values(X)
                # [:, :, 1] selecte les valeurs SHAP pour la classe 1 = Phishing
                # [0] prend la premiere (et unique) ligne (notre site)
                shap_vals = sv_site[:, :, 1][0]

                # on associe chaque nom de feature a sa valeur SHAP
                feature_impact = list(zip(feature_names, shap_vals))
                # on trie par valeur absolue decroissante -> les plus influentes en premier
                feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
                top5 = feature_impact[:5]  # on garde seulement le top 5

                st.subheader("Pourquoi ce resultat ?")
                for feat, val in top5:
                    # val > 0 = cette feature pousse vers Phishing
                    # val < 0 = cette feature pousse vers Legitime
                    sens = "augmente" if val > 0 else "diminue"
                    st.write(f"{sens} le risque : {feat} (impact: {val:+.3f})")

                # Section repliable pour voir toutes les features
                with st.expander("Voir toutes les features detectees"):
                    # .T = transpose (lignes <-> colonnes) pour afficher verticalement
                    st.dataframe(X.T.rename(columns={0: 'valeur'}))

                # ETAPE 8 : bouton feedback (apprentissage continu)
                st.divider()
                st.subheader("Cette prediction est-elle correcte ?")
                # st.columns(2) cree 2 colonnes cote a cote
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Oui, c'est correct"):
                        st.success("Merci pour votre retour !")

                with col2:
                    if st.button("Non, signaler une erreur"):
                        # si le modele a dit PHISHING, la vraie reponse est LEGITIME et vice versa
                        label_correct = "LEGITIME" if label == "PHISHING" else "PHISHING"
                        sauvegarder_correction(url_input, label, label_correct)
                        nb = compter_corrections()
                        st.warning(f"Correction enregistree ! ({nb} corrections au total)")
                        if nb >= 10:
                            # seuil atteint -> on peut reentainer le modele
                            st.info("Seuil de 10 corrections atteint - lancer le reentainement : python notebooks/03_retrain.py")
        else:
            st.warning("Merci d'entrer une URL.")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.title("Dashboard Analytique")

    # Charge toutes les analyses depuis la base SQLite
    df_hist = charger_historique()

    if df_hist.empty:
        # Aucune analyse encore -> message d'information
        st.info("Aucune analyse enregistree. Va sur la page Analyseur pour commencer !")
    else:
        # st.columns(3) cree 3 colonnes cote a cote pour les metriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total analyses", len(df_hist))
        with col2:
            # compte le nombre de lignes ou label == 'PHISHING'
            nb_phishing = (df_hist['label'] == 'PHISHING').sum()
            st.metric("Phishing detectes", nb_phishing)
        with col3:
            taux = nb_phishing / len(df_hist) * 100
            st.metric("Taux de phishing", f"{taux:.1f}%")

        st.divider()

        # Graphique en barres : nombre de PHISHING vs LEGITIME
        st.subheader("Repartition des resultats")
        repartition = df_hist['label'].value_counts()  # compte chaque label
        st.bar_chart(repartition)

        # Graphique en ligne : evolution de la probabilite de phishing dans le temps
        st.subheader("Evolution du risque au fil des analyses")
        df_hist_sorted = df_hist.sort_values('id')  # tri par ordre chronologique
        st.line_chart(df_hist_sorted.set_index('id')['proba'])
        # set_index('id') utilise l'id comme axe X
        # ['proba'] selectionne uniquement la colonne des probabilites pour l'axe Y


# ══════════════════════════════════════════════════════════════
# PAGE 3 — HISTORIQUE
# ══════════════════════════════════════════════════════════════
elif page == "Historique":
    st.title("Historique des analyses")

    df_hist = charger_historique()

    if df_hist.empty:
        st.info("Aucune analyse enregistree pour le moment.")
    else:
        # Menu deroulant pour filtrer par type de resultat
        filtre = st.selectbox("Filtrer par resultat :", ["Tous", "PHISHING", "LEGITIME"])

        if filtre != "Tous":
            # garde seulement les lignes dont le label correspond au filtre
            df_affiche = df_hist[df_hist['label'] == filtre]
        else:
            df_affiche = df_hist  # pas de filtre -> on affiche tout

        # Copie pour ne pas modifier le DataFrame original
        df_affiche_display = df_affiche.copy()

        # Formate la probabilite en pourcentage lisible (ex: 0.897 -> "89.7%")
        df_affiche_display['proba'] = df_affiche_display['proba'].apply(lambda x: f"{x:.1%}")

        # Renomme les colonnes pour un affichage plus propre
        df_affiche_display = df_affiche_display.rename(columns={
            'url': 'URL',
            'label': 'Resultat',
            'proba': 'Confiance',
            'date': 'Date'
        })

        # Affiche le tableau interactif
        # use_container_width=True : le tableau prend toute la largeur disponible
        st.dataframe(
            df_affiche_display[['URL', 'Resultat', 'Confiance', 'Date']],
            use_container_width=True
        )
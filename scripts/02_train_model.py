# Ce script entraine les modeles de Machine Learning sur le dataset nettoye.
# Il produit 3 fichiers sauvegardes dans le dossier models/ :
#   - ensemble_model.pkl : le modele entraine (Random Forest + XGBoost combines)
#   - scaler.pkl         : le StandardScaler entraine (normalisation des features)
#   - feature_names.pkl  : la liste ordonnee des 30 noms de features
#
# A lancer depuis la racine du projet :
# python scripts/02_train_model.py
#
# Resultat attendu : AUC-ROC > 0.99 sur les 3 modeles

import pandas as pd   # chargement et manipulation du dataset
from sklearn.model_selection import train_test_split        # separation train/test
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # modeles sklearn
from sklearn.preprocessing import StandardScaler            # normalisation des features
from sklearn.metrics import classification_report, roc_auc_score  # metriques d'evaluation
from xgboost import XGBClassifier  # algorithme XGBoost (bibliotheque separee de sklearn)
import joblib  # sauvegarde et chargement des modeles ML sur le disque
import os      # creation de dossiers


# ── 1. CHARGER LES DONNEES ────────────────────────────────────
# On charge le dataset nettoye produit par 01_exploration.py
# (colonne Index supprimee, 11054 lignes, 31 colonnes)
df = pd.read_csv('data/processed/dataset_clean.csv')

# Separation des features (X) et du label cible (y)
# X = les 30 colonnes qui servent a predire (UsingIP, LongURL, HTTPS...)
# y = la colonne 'class' qui contient la vraie reponse (1=Phishing, -1=Legitime)
X = df.drop(columns=['class'])
y = df['class']

# Conversion des labels : -1 devient 0
# XGBoost exige des labels 0 et 1 (pas -1 et 1)
# Apres conversion : 0 = Legitime, 1 = Phishing
y = y.replace({-1: 0})

print(f"Dataset charge : {X.shape[0]} lignes, {X.shape[1]} features")
# .shape[0] = nombre de lignes, .shape[1] = nombre de colonnes


# ── 2. SEPARER TRAIN / TEST ───────────────────────────────────
# On divise le dataset en deux parties :
#   - X_train, y_train : 80% des donnees -> le modele apprend dessus
#   - X_test, y_test   : 20% des donnees -> on evalue le modele dessus
#
# Pourquoi separer ?
# Si on evaluait sur les memes donnees qu'on a utilisees pour apprendre,
# le modele aurait un score parfait mais serait inutilisable sur de nouvelles URLs.
# Le test set simule des donnees jamais vues.
#
# random_state=43 : fixe le point de depart du generateur aleatoire
# -> les resultats sont identiques a chaque execution (reproductibilite)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=43
)

print(f"Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
# Resultat attendu : Train 8843 lignes, Test 2211 lignes


# ── 3. NORMALISER LES FEATURES ────────────────────────────────
# Le StandardScaler normalise chaque feature pour qu'elle ait :
#   - une moyenne de 0
#   - un ecart-type de 1
# Formule : Vn = (V - moyenne) / ecart_type
#
# Pourquoi normaliser ?
# Certaines features ont des echelles tres differentes :
#   PageRank : entre 0 et 1
#   longueur URL : entre 10 et 200
# Sans normalisation, les features avec de grandes valeurs dominent le modele.

scaler = StandardScaler()

# .fit_transform(X_train) fait deux choses :
#   .fit()      : calcule la moyenne et l'ecart-type de chaque colonne sur X_train
#   .transform(): applique la formule de normalisation
X_train_scaled = scaler.fit_transform(X_train)

# .transform(X_test) : applique UNIQUEMENT la normalisation (pas de fit)
# On reutilise la moyenne et l'ecart-type calcules sur X_train
# (car en production on ne connait pas les stats du jeu de test)
X_test_scaled = scaler.transform(X_test)


# ── 4. ENTRAINER LES MODELES ──────────────────────────────────

# -- Random Forest --
# Un Random Forest est un ensemble de 200 arbres de decision independants.
# Chaque arbre apprend sur un sous-ensemble aleatoire des donnees et des features.
# La prediction finale est le vote majoritaire de tous les arbres.
# Avantage : robuste au bruit, peu sensible au surapprentissage (overfitting).
print("Entrainement Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,  # nombre d'arbres de decision dans la foret
    random_state=43    # fixe le hasard pour la reproductibilite
)
# .fit() = la phase d'apprentissage
# On montre au modele les 8843 sites avec leurs vraies etiquettes
# Le modele apprend les regles qui distinguent Phishing de Legitime
rf.fit(X_train_scaled, y_train)
print("Random Forest entraine")


# -- XGBoost --
# XGBoost (Extreme Gradient Boosting) entraine les arbres EN SEQUENCE.
# Chaque nouvel arbre corrige les erreurs du precedent.
# C'est l'algorithme gagnant de nombreuses competitions Kaggle.
# Plus lent a entrainer mais souvent plus precis que Random Forest.
print("Entrainement XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,      # nombre d'arbres boostes
    learning_rate=0.05,    # taux d'apprentissage : chaque arbre corrige lentement
                           # (lent mais plus precis qu'un learning_rate eleve)
    random_state=42,
    eval_metric='logloss'  # metrique interne de XGBoost (evite un warning)
)
xgb.fit(X_train_scaled, y_train)
print("XGBoost entraine")


# ── 5. ENSEMBLE (VOTING) ──────────────────────────────────────
# Un VotingClassifier combine plusieurs modeles en un seul.
# voting='soft' : chaque modele donne une probabilite (pas juste 0 ou 1)
#                 et on fait la moyenne des probabilites -> plus precis
# voting='hard' : chaque modele vote 0 ou 1 et on prend la majorite -> moins precis
#
# Analogie : comme un jury de 2 experts qui se concertent
# au lieu d'un seul expert qui decide seul.
print("Entrainement Ensemble...")
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],  # liste des modeles a combiner avec leurs noms
    voting='soft'
)
# .fit() re-entraine les deux modeles ensemble ET configure le systeme de vote
ensemble.fit(X_train_scaled, y_train)
print("Ensemble entraine")


# ── 6. EVALUER LES MODELES ────────────────────────────────────
# On evalue les 3 modeles (RF, XGBoost, Ensemble) sur le jeu de test
# pour comparer leurs performances et choisir le meilleur.
print("\n=== RESULTATS ===")
for nom, modele in [('Random Forest', rf), ('XGBoost', xgb), ('Ensemble', ensemble)]:

    # .predict() retourne la classe predite pour chaque site du test set
    # Resultat : tableau de 0 et 1 -> ex: [0, 1, 1, 0, 1, ...]
    y_pred = modele.predict(X_test_scaled)

    # .predict_proba() retourne les probabilites pour chaque classe
    # Resultat : tableau de [proba_legitime, proba_phishing] par site
    # [:, 1] recupere uniquement la colonne proba_phishing pour tous les sites
    y_proba = modele.predict_proba(X_test_scaled)[:, 1]

    # AUC-ROC : mesure la capacite du modele a distinguer les deux classes
    # 1.0 = parfait, 0.5 = aleatoire (comme pile ou face)
    # Notre objectif : AUC > 0.97
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {nom} ---")

    # classification_report affiche un tableau complet avec :
    #   precision : parmi les URLs classees Phishing, combien sont vraiment Phishing ?
    #   recall    : parmi les vraies URLs Phishing, combien ont ete detectees ?
    #   f1-score  : moyenne harmonique de precision et recall
    #   support   : nombre de sites dans chaque classe dans le test set
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC : {auc:.4f}")


# ── 7. SAUVEGARDER ───────────────────────────────────────────
# On sauvegarde 3 fichiers essentiels pour l'interface Streamlit :
os.makedirs('models', exist_ok=True)

# Le modele entraine : contient tous les arbres appris
joblib.dump(ensemble, 'models/ensemble_model.pkl')

# Le scaler entraine : contient les moyennes et ecarts-types calcules sur X_train
# Indispensable pour normaliser de nouvelles URLs de la meme facon qu'a l'entrainement
joblib.dump(scaler, 'models/scaler.pkl')

# La liste ordonnee des noms de features
# Indispensable pour construire le vecteur de features dans le BON ORDRE
# lors de l'analyse d'une nouvelle URL dans Streamlit
joblib.dump(list(X.columns), 'models/feature_names.pkl')

print("\nModele sauvegarde dans models/ensemble_model.pkl")
print("Scaler sauvegarde dans models/scaler.pkl")
print("Noms des features sauvegardes dans models/feature_names.pkl")
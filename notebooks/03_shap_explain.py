# Ce script calcule et visualise les valeurs SHAP du modele entraine.
# SHAP (SHapley Additive exPlanations) permet d'expliquer chaque prediction
# en montrant la contribution de chaque feature a la decision finale.
#
# Ce script produit 3 sorties :
#   - reports/shap_summary.png   : graphique global des features les plus importantes
#   - reports/shap_waterfall.png : explication detaillee d'un site specifique
#   - models/shap_explainer.pkl  : l'explainer sauvegarde pour l'interface Streamlit
#
# A lancer depuis la racine du projet APRES 02_train_model.py :
# python notebooks/03_shap_explain.py

import pandas as pd   # manipulation des donnees
import numpy as np    # operations sur les tableaux numeriques
import shap           # bibliotheque d'explicabilite des modeles ML
import joblib         # chargement des modeles sauvegardes

# matplotlib.use('Agg') : force matplotlib a sauvegarder les graphiques
# sans les afficher a l'ecran (necessaire sur les serveurs sans interface graphique
# comme le terminal Linux de Crostini)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── 1. CHARGER LE MODELE ET LES DONNEES ──────────────────────
print("Chargement du modele...")

# Charge le modele ensemble entraine a l'etape 4 (RF + XGBoost combines)
ensemble = joblib.load('models/ensemble_model.pkl')

# Charge le StandardScaler entraine a l'etape 4
# Indispensable pour normaliser les donnees de la meme facon qu'a l'entrainement
scaler = joblib.load('models/scaler.pkl')

# Charge la liste ordonnee des 30 noms de features
# pour que les graphiques SHAP affichent les vrais noms (UsingIP, LongURL...)
# au lieu de simples numeros de colonnes
feature_names = joblib.load('models/feature_names.pkl')

# Charge le dataset nettoye pour calculer les valeurs SHAP
df = pd.read_csv('data/processed/dataset_clean.csv')

# Separation features / labels
X = df.drop(columns=['class'])   # les 30 features -> ce qu'on donne au modele
y = df['class'].replace({-1: 0}) # les vrais labels -> 0=Legitime, 1=Phishing

# Normalisation du dataset complet avec le scaler deja entraine
# .transform() uniquement (pas .fit_transform()) car on reutilise les stats de l'entrainement
X_scaled = scaler.transform(X)

# Conversion en DataFrame pour conserver les noms de colonnes
# (necessaire pour que SHAP affiche les noms des features sur les graphiques)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Echantillon de 500 sites pris aleatoirement parmi les 11054
# Calcul SHAP sur 11054 sites serait trop lent (plusieurs minutes)
# 500 sites suffisent pour avoir une vue representative et fiable
X_sample = X_scaled_df.sample(500, random_state=43)
print(f"Echantillon : {X_sample.shape}")
# Resultat attendu : (500, 30) -> 500 sites, 30 features


# ── 2. CREER L'EXPLAINER SHAP ─────────────────────────────────
print("\nCreation de l'explainer SHAP...")

# On extrait uniquement le Random Forest de l'ensemble
# car TreeExplainer ne fonctionne qu'avec des modeles a base d'arbres individuels
# Il ne sait pas "ouvrir" un VotingClassifier directement
rf_model = ensemble.named_estimators_['rf']

# TreeExplainer analyse la structure interne de tous les 200 arbres du Random Forest
# pour calculer mathematiquement la contribution exacte de chaque feature
# (via la theorie des jeux : valeurs de Shapley)
# Beaucoup plus rapide que KernelExplainer (generique mais lent)
explainer = shap.TreeExplainer(rf_model)
print("Explainer cree")


# ── 3. CALCULER LES VALEURS SHAP ─────────────────────────────
print("\nCalcul des valeurs SHAP (peut prendre 1-2 min)...")

# .shap_values() calcule pour chaque site et chaque feature
# sa contribution a la prediction finale (positive = pousse vers Phishing,
# negative = pousse vers Legitime)
shap_values = explainer.shap_values(X_sample)

print("Valeurs SHAP calculees")

# np.array(shap_values).shape affiche les dimensions du resultat
# Resultat attendu : (500, 30, 2)
#   500 = nb de sites dans l'echantillon
#    30 = nb de features
#     2 = nb de classes (0=Legitime, 1=Phishing)
print(f"Shape shap_values : {np.array(shap_values).shape}")


# ── 4. GRAPHIQUE 1 : SUMMARY PLOT ────────────────────────────
# Le summary plot montre les features les plus importantes sur l'ensemble des 500 sites.
# Chaque point = un site. Position horizontale = impact sur la prediction.
# Couleur = valeur de la feature (rouge = haute, bleu = basse).
# Les features sont triees de la plus importante (haut) a la moins importante (bas).
print("\nGeneration du graphique 1 - Summary Plot...")

plt.figure()  # cree une nouvelle figure vierge

shap.summary_plot(
    shap_values[:, :, 1],  # valeurs SHAP pour la classe 1 = Phishing uniquement
    X_sample,              # les vraies valeurs des features (pour la couleur des points)
    feature_names=feature_names,  # noms des features a afficher
    show=False             # ne pas afficher a l'ecran -> on va sauvegarder
)

plt.title("Top features - Impact sur la detection phishing")
plt.tight_layout()  # ajuste les marges pour que rien ne soit coupe
# dpi=150 : resolution de l'image (plus haute = plus nette)
# bbox_inches='tight' : evite que le titre ou les labels soient coupes
plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()  # ferme la figure pour liberer la memoire
print("Sauvegarde : reports/shap_summary.png")


# ── 5. GRAPHIQUE 2 : WATERFALL PLOT ──────────────────────────
# Le waterfall plot explique la prediction d'UN SEUL site en detail.
# Il montre comment chaque feature fait monter ou descendre la probabilite
# depuis la valeur de base (moyenne) jusqu'a la prediction finale.
# C'est ce graphique qui est utilise dans l'interface Streamlit pour
# expliquer chaque analyse en temps reel.
print("\nGeneration du graphique 2 - Waterfall Plot...")

# explainer(X_sample) retourne un objet Explanation (plus riche que shap_values)
# qui contient aussi la valeur de base et les noms de features -> necessaire pour waterfall
sv = explainer(X_sample)

# [:, :, 1] selectionne les valeurs SHAP pour la classe 1 = Phishing
sv_phishing = sv[:, :, 1]

plt.figure()
# sv_phishing[0] = explication du premier site de l'echantillon
shap.plots.waterfall(sv_phishing[0], show=False)
plt.title("Explication detaillee - Site 1 (classe Phishing)")
plt.tight_layout()
plt.savefig('reports/shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Sauvegarde : reports/shap_waterfall.png")


# ── 6. EXPLICATION TEXTUELLE AUTOMATIQUE ─────────────────────
# Cette fonction transforme les valeurs SHAP numeriques en phrases lisibles.
# C'est l'originalite principale du projet : expliquer en francais POURQUOI
# le modele a classe une URL comme phishing ou legitime.
# Ex: "PHISHING a 94% car -> 'HTTPS' augmente le risque (impact: +0.420)"
print("\n--- EXPLICATION AUTOMATIQUE ---")

def expliquer_prediction(index):
    # Recupere une seule ligne du DataFrame (un seul site)
    # iloc[index:index+1] retourne un DataFrame 2D (1 ligne)
    # au lieu de iloc[index] qui retourne une Series 1D
    # -> Le modele ML et SHAP attendent toujours un tableau 2D
    site = X_sample.iloc[index:index+1]

    # Recupere la probabilite que ce site soit du phishing
    # predict_proba retourne [[proba_legitime, proba_phishing]]
    # [0][1] = premiere ligne, deuxieme element = proba_phishing
    proba = ensemble.predict_proba(site)[0][1]
    label = "PHISHING" if proba > 0.5 else "LEGITIME"

    # Calcule les valeurs SHAP pour ce site uniquement
    # (meme logique que l'etape 3 mais sur 1 site au lieu de 500)
    sv_site = explainer.shap_values(site)
    # [:, :, 1] = classe Phishing, [0] = premiere (et unique) ligne
    # Resultat : tableau de 30 valeurs, une par feature
    shap_vals = sv_site[:, :, 1][0]

    # Association nom de feature <-> valeur SHAP
    # zip() couple chaque nom avec sa valeur correspondante
    feature_impact = list(zip(feature_names, shap_vals))

    # Tri par valeur absolue decroissante
    # abs() car une valeur de -0.30 est aussi importante que +0.30
    # (meme impact mais dans le sens oppose)
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    # On garde seulement le top 3 des features les plus influentes
    top3 = feature_impact[:3]

    print(f"\nSite analyse (index {index}) :")
    print(f"  Resultat  : {label}")
    # :.1% formate 0.94 en "94.0%"
    print(f"  Confiance : {proba:.1%}")
    print(f"  Pourquoi ?")
    for feat, val in top3:
        # val > 0 : cette feature pousse vers Phishing (augmente le risque)
        # val < 0 : cette feature pousse vers Legitime (diminue le risque)
        sens = "augmente" if val > 0 else "diminue"
        # :+.3f formate toujours avec le signe (+ ou -) et 3 decimales
        print(f"    {feat} {sens} le risque (impact: {val:+.3f})")

# Test sur 3 sites differents de l'echantillon
for i in [0, 1, 2]:
    expliquer_prediction(i)


# ── 7. SAUVEGARDER L'EXPLAINER ────────────────────────────────
# L'explainer est sauvegarde pour etre reutilise directement dans app.py
# sans avoir a le recalculer a chaque demarrage de l'interface
# (le recalcul prendrait plusieurs minutes a chaque lancement)
joblib.dump(explainer, 'models/shap_explainer.pkl')
print("\nExplainer SHAP sauvegarde dans models/shap_explainer.pkl")
print("\nETAPE SHAP TERMINEE !")
print("Graphiques disponibles dans reports/")
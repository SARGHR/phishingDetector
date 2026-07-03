import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ── 1. CHARGER LE MODÈLE ET LES DONNÉES ──────────────────────
print("Chargement du modele...")
ensemble = joblib.load('models/ensemble_model.pkl') #modele entrainer a E4
scaler   = joblib.load('models/scaler.pkl')#le StandardScaler entraîné à l'étape 4 (normaliser les Data )
feature_names = joblib.load('models/feature_names.pkl') #Recharge la liste des 30 noms de features 

df = pd.read_csv('data/processed/dataset_clean.csv')
X  = df.drop(columns=['class']) #separer les features X
y  = df['class'].replace({-1: 0})#separer les labels Y

# Normaliser
X_scaled = scaler.transform(X)#normaliser le dataset(moyenne=0,ecartType=1)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Prendre un échantillon de 500 pour aller plus vite
X_sample = X_scaled_df.sample(500, random_state=43) #prendre 500/11054 sites 
print(f"Echantillon : {X_sample.shape}")

# ── 2. CRÉER L'EXPLAINER SHAP ─────────────────────────────────
print("\nCreation de l'explainer SHAP...")
rf_model = ensemble.named_estimators_['rf'] #(rappel esemble combine les 2 model rf et xgb) "named.estimator(rf) permet d'extraire que le modele rf"
explainer = shap.TreeExplainer(rf_model) #creer l'explainer
print(" Explainer cree")

# ── 3. CALCULER LES VALEURS SHAP ─────────────────────────────
print("\nCalcul des valeurs SHAP (peut prendre 1-2 min)... merci de patienter")
shap_values = explainer.shap_values(X_sample) #permet de connaitre le poid de chaque feature
print("✅ Valeurs SHAP calculées")
print(f"Shape shap_values : {np.array(shap_values).shape}")

# ── 4. GRAPHIQUE 1 — FEATURES LES PLUS IMPORTANTES ───────────
print("\nGénération du graphique 1 - Summary Plot...")
plt.figure()
shap.summary_plot(
    shap_values[:, :, 1],
    X_sample,
    feature_names=feature_names,
    show=False
)
plt.title("Top features — Impact sur la détection phishing")
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Sauvegardé : reports/shap_summary.png")

# ── 5. GRAPHIQUE 2 — EXPLICATION D'UN SITE SPÉCIFIQUE ─────────
print("\nGénération du graphique 2 - Waterfall Plot...")
sv = explainer(X_sample)
# [:, :, 1] → on sélectionne la classe 1 = PHISHING
sv_phishing = sv[:, :, 1]
plt.figure()
shap.plots.waterfall(sv_phishing[0], show=False)
plt.title("Explication détaillée — Site 1 (classe Phishing)")
plt.tight_layout()
plt.savefig('reports/shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Sauvegardé : reports/shap_waterfall.png")

# ── 6. EXPLICATION TEXTUELLE AUTOMATIQUE ──────────────────────
#transformation des valeurs shape en phrase 
print("\n--- EXPLICATION AUTOMATIQUE EN FRANÇAIS ---")

def expliquer_prediction(index):
    site = X_sample.iloc[index:index+1] # recup une ligne du dataFrame X_sample  ala pos index 
                          #le shape et ml prennent un tab de 2D
    # Prédiction
    proba = ensemble.predict_proba(site)[0][1]  #recup la proba que le site soit du phishing sous forme d'un seule chiffre
    label = "PHISHING" if proba > 0.5 else "LÉGITIME" # regle de decision classique

    # Valeurs SHAP pour ce site — classe 1 = phishing
    sv_site = explainer.shap_values(site) #meme que etape 3 mais ici pour un seule site 
    shap_vals = sv_site[:, :, 1][0]
    #res : tab de 30 val ,une par feature

    # Top 3 features qui ont le plus influencé
    feature_impact = list(zip(feature_names, shap_vals))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    top3 = feature_impact[:3]

    print(f"\nSite analysé (index {index}) :")
    print(f"  Résultat    : {label}")
    print(f"  Confiance   : {proba:.1%}")
    print(f"  Pourquoi ?")
    for feat, val in top3:
        sens = "augmente" if val > 0 else "diminue"
        print(f"    → '{feat}' {sens} le risque (impact: {val:+.3f})")

# Analyser 3 sites différents
for i in [0, 1, 2]:
    expliquer_prediction(i)

# ── 7. SAUVEGARDER L'EXPLAINER ────────────────────────────────
joblib.dump(explainer, 'models/shap_explainer.pkl')
print("\n✅ Explainer SHAP sauvegardé dans models/shap_explainer.pkl")
print("\n🎉 ÉTAPE SHAP TERMINÉE !")
print("Graphiques disponibles dans reports/")
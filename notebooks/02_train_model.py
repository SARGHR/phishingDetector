import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os


# 1. CHARGER LES DONNÉES
df = pd.read_csv('data/processed/dataset_clean.csv')

# Séparer features (X) et label (y)
X = df.drop(columns=['class'])  #les colonnes qui servent a predir
y = df['class'] #les colonnes cible 

# Convertir les labels : -1 → 0 (XGBoost a besoin de 0 et 1, pas -1 et 1)
y = y.replace({-1: 0})
    #shape return les dim d'un tab en un tuple(l,c)
print(f" Dataset charge : {X.shape[0]} lignes, {X.shape[1]} features")

# =====================
# 2. SEPARER TRAIN / TEST
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% pour tester, 80% pour entraîner
    random_state=43     # fixe le hasard → résultats reproductibles
)

print(f" Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")

# =====================
# 3. NORMALISER LES FEATURES
# =====================
scaler = StandardScaler() #creer un objet StandarScaler
X_train_scaled = scaler.fit_transform(X_train)# (.fit) sert a memoriser et observer le scaler puis (transform) sert a appliquee la formule de normalisation(Vn=(V-M)/s) s: ecart type  
X_test_scaled = scaler.transform(X_test)
# Pourquoi ? Certaines features ont des valeurs très différentes
# ex: PageRank entre 0 et 10, longueur URL entre 0 et 200
# La normalisation met tout à la même échelle

# =====================
# 4. ENTRAINER LES MODELES
# =====================
print("\n Entrainement Random Forest...")
rf = RandomForestClassifier(  #creation d'objet RFC et le stocker dans rf
    n_estimators=200,   # le nb (200) d'arbres de decision
    random_state=43  #Fixe le point de départ du hasard.

)

#rf.fit(X_train_scaled, y_train) = "Montre les 8 843 sites d'entraînement avec leurs réponses correctes aux 200 arbres pour qu'ils apprennent à distinguer le phishing du légitime
rf.fit(X_train_scaled, y_train) # .fit c'est <=> fct apprendre en ML
        #les features , les labels
        #ex(question) , (reponse)         
print(" Random Forest entraine")


print("\n Entraînement XGBoost...")
xgb = XGBClassifier(
    n_estimators=300,       # 300 arbres boostés
    learning_rate=0.05,     # la vitesse d'apprentissage (lentement mais surement)
    random_state=42,
    eval_metric='logloss'   # evite un warning
)
xgb.fit(X_train_scaled, y_train)
print(" XGBoost entraîné")

# =====================
# 5. ENSEMBLE (VOTING)
# =====================
print("\n Entrainement Ensemble...")
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],#liste des models a combiner
    voting='soft'   # utilise les probabilités, plus précis que hard voting
)
ensemble.fit(X_train_scaled, y_train) # reentrainer re,xgb derniere fois  + config de systeme de vote entre eux
print("Ensemble entraine")

# =====================
# 6. ÉVALUER LES MODÈLES
# =====================
print("\n=== RÉSULTATS ===")
for nom, modele in [('Random Forest', rf), ('XGBoost', xgb), ('Ensemble', ensemble)]:
    y_pred = modele.predict(X_test_scaled) #regarder l'etat de chaque site(legitime/phishing) en gros il vas faire un tableau d'etat T=[0,1,1,0,...,1]
    y_proba = modele.predict_proba(X_test_scaled)[:, 1] # meme chose su'avant sauf cette fois ci des proba(0<p<1) au lieu de (1,0) il recupere uniquement la colonne qui contient la proba de phishing 
    auc = roc_auc_score(y_test, y_proba) #auc-roc mesure si le model classe bien le ph avec les probaH et les leg avec probaB
    print(f"\n--- {nom} ---")
    print(classification_report(y_test, y_pred))# un tab de mytrique complet
    print(f"AUC-ROC : {auc:.4f}")

# =====================
# 7. SAUVEGARDER
# =====================
os.makedirs('models', exist_ok=True)
joblib.dump(ensemble, 'models/ensemble_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(list(X.columns), 'models/feature_names.pkl')
print("\nModèle sauvegardé dans models/ensemble_model.pkl")
print(" Scaler sauvegardé dans models/scaler.pkl")
print(" Noms des features sauvegardés dans models/feature_names.pkl")
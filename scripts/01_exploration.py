# Ce script est la premiere etape du projet.
# Son role : charger le dataset brut, verifier sa qualite,
# le nettoyer et sauvegarder une version propre prete pour le ML.
# Regle d'or en data science : on ne modifie jamais les donnees brutes,
# on travaille toujours sur une copie nettoyee.
#
# A lancer UNE SEULE FOIS depuis la racine du projet :
# python scripts/01_exploration.py

import pandas as pd          # pour charger et manipuler les donnees
import matplotlib.pyplot as plt  # pour creer et sauvegarder des graphiques
import os                    # pour creer des dossiers si necessaire


# ── CHARGEMENT DU DATASET ─────────────────────────────────────
# pd.read_csv lit le fichier CSV et le transforme en DataFrame
# Un DataFrame est un tableau comme Excel : lignes = sites web, colonnes = features
df = pd.read_csv('data/raw/phishing.csv')


# ── EXPLORATION ───────────────────────────────────────────────

print("=== SHAPE ===")
# .shape retourne un tuple (nb_lignes, nb_colonnes)
# Permet de verifier qu'on a assez de donnees pour entrainer un bon modele
print(df.shape)
# Resultat attendu : (11054, 32) -> 11054 sites, 32 colonnes

print("\n=== TYPES ===")
# .dtypes affiche le type de chaque colonne (int64, float64, object...)
# On verifie que toutes les colonnes sont numeriques (obligatoire pour le ML)
print(df.dtypes)
# Resultat attendu : toutes les colonnes sont en int64

print("\n=== VALEURS MANQUANTES ===")
# .isnull() cree un tableau de True/False (True = valeur manquante)
# .sum() additionne les True par colonne (True = 1, False = 0)
# .sum() encore additionne tout -> nombre total de valeurs manquantes
print(df.isnull().sum().sum())
# Resultat attendu : 0 -> dataset parfaitement propre, rien a corriger

print("\n=== REPARTITION DES CLASSES ===")
# .value_counts() compte le nombre d'occurrences de chaque valeur unique
# dans la colonne 'class' (qui contient 1 = Phishing ou -1 = Legitime)
# Permet de verifier si les deux classes sont equilibrees
# (sinon le modele aurait tendance a tout predire comme la classe majoritaire)
print(df['class'].value_counts())
# Resultat attendu :
#  1    6157  (55.7%) -> Phishing
# -1    4897  (44.3%) -> Legitime
# -> Bien equilibre, pas de correction necessaire


# ── NETTOYAGE ─────────────────────────────────────────────────

# Suppression de la colonne 'Index' qui ne contient qu'un numero de ligne
# Cette colonne est inutile pour le ML et pourrait perturber le modele
# (le modele pourrait apprendre a se baser sur le numero de ligne, ce qui n'a aucun sens)
df = df.drop(columns=['Index'])
# Apres suppression : shape passe de (11054, 32) a (11054, 31)


# ── SAUVEGARDE DE LA VERSION PROPRE ───────────────────────────

# Cree le dossier data/processed/ s'il n'existe pas
# exist_ok=True evite une erreur si le dossier existe deja
os.makedirs('data/processed', exist_ok=True)

# Sauvegarde le DataFrame nettoye en CSV
# index=False evite d'ecrire une colonne supplementaire de numeros de lignes
df.to_csv('data/processed/dataset_clean.csv', index=False)
print("\nSauvegarde dans data/processed/dataset_clean.csv")


# ── GRAPHIQUE DE REPARTITION ──────────────────────────────────

# Cree le dossier reports/ s'il n'existe pas
os.makedirs('reports', exist_ok=True)

# .value_counts() compte les occurrences de chaque classe
# .plot(kind='bar') cree un graphique en barres
# color=[...] colorie les barres (bleu = Phishing, rouge = Legitime)
df['class'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])

# Ajoute un titre au graphique
plt.title('Repartition Phishing vs Legitime')

# Sauvegarde le graphique en image PNG dans le dossier reports/
# Sans savefig() le graphique s'afficherait a l'ecran mais ne serait pas sauvegarde
plt.savefig('reports/distribution_classes.png')
print("Graphique sauvegarde dans reports/distribution_classes.png")
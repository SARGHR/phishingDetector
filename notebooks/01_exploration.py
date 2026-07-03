import pandas as pd
import matplotlib.pyplot as plt
import os

# Charger le dataset
#lire le fichier phishing.csv  et le transformer en tab excel
df = pd.read_csv('data/raw/phishing.csv')

# Explorer
print("=== SHAPE ===")
print("le nbr de lignes et de colonnes ")
#affiche nb:lignes ,nb:colonnes
print(df.shape)


print("\n=== TYPES ===")
print("affiche le type de chaque colonnes")
print(df.dtypes)

print("\n=== VALEURS MANQUANTES ===")
print("verif si valeurs manquantes")
print(df.isnull().sum().sum())

print("\n=== RÉPARTITION DES CLASSES ===")
print("compte le nbr totale de chaque valeurs(dans notre cas 1 et -1)")
print(df['class'].value_counts())

# Supprimer colonne inutile
df = df.drop(columns=['Index'])

# Sauvegarder version propre
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/dataset_clean.csv', index=False)
print("\n✅ Sauvegardé dans data/processed/dataset_clean.csv")

# Graphique
os.makedirs('reports', exist_ok=True)
df['class'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Répartition Phishing vs Légitime')
plt.savefig('reports/distribution_classes.png')
print("✅ Graphique sauvegardé")
# Prédiction des prix immobiliers avec Machine Learning

## Description du projet
L’objectif de ce projet est de prédire, via l’apprentissage automatique, le prix de vente d’une maison ou d’un appartement en fonction de sa ville, sa surface, son diagnostique de performance énergétique, etc.
Il utilise l'apprentissage automatique et s'appuie sur les annonces du site [Immo Entre Particuliers](https://www.immo-entre-particuliers.com/).
Le projet suit trois jalons principaux :
1. Extraction et stockage des données.
2. Nettoyage et pré-traitement des données.
3. Entraînement et évaluation de modèles de prédiction.


## Prérequis
Avant d'utiliser ce projet, assurez-vous d'avoir installé les logiciels et bibliothèques suivants :

- Python 3.8 ou version ultérieure
- `beautifulsoup4`
- `matplotlib`
- `numpy`
- `pandas`
- `requests`
- `scikit-learn`
- `seaborn`

## Installation
1. Clonez le dépôt :
   ```bash
   git clone https://github.com/ANGELlNE/projet_s6.git
   cd projet_s6
   ```
2. Créez et activez un environnement virtuel :
   ```bash
   python -m venv env
   source env/bin/activate  # Sur Mac/Linux
   env\Scripts\activate  # Sur Windows
   ```
3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
1. Exécutez le script principal pour entraîner et évaluer les modèles :
   ```bash
   python projet.py
   ```
2. Pour explorer les données, utilisez le notebook Jupyter fourni :
   ```bash
   jupyter notebook projet.ipynb
   ```

## Roadmap
- [x] Extraction et stockage des données
- [x] Nettoyage et prétraitement
- [x] Implémentation des modèles de régression
- [ ] Optimisation du code

## Modèles de Machine Learning
Nous utilisons plusieurs modèles pour prédire les prix immobiliers :
- **Régression linéaire**
- **Arbres de décision**
- **K plus proches voisins (KNN)**

## Contributeurs
- **N.Ngoc Uyen Chi** - [GitHub](https://github.com/nguyenngocuyenchi) *(uyenchi.nguyenngoc04@gmail.com)*
- **T.Angéline** - [GitHub](https://github.com/ANGELlNE) *(tamilangeline@yahoo.com)*

Pour toute question ou suggestion, n'hésitez pas à nous contacter ou à ouvrir une issue sur le dépôt GitHub.

Merci de votre intérêt pour notre projet !
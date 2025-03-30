#########################################################################################
#                                                                                       #
#                 Projet de prédiction, via l’apprentissage automatique :               #
#                                                                                       #
#########################################################################################

from bs4 import BeautifulSoup
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#################################################################
#       Premier jalon : récupération des données en python      #
#################################################################


# Q1
def getsoup(url):
    return BeautifulSoup(requests.get(url).text,'html.parser')


# Q2
class NonValide(Exception):
    pass


# Q3
def prix(soup):
    try:
        prix_tag=soup.find("p", class_="product-price")
        prix_text=prix_tag.text.replace("€", "").replace(" ", "").strip()
        prix=int(prix_text)
        if prix<10000:
            raise NonValide("Prix trop bas.")
        return prix
    except:
        raise NonValide("Prix non trouvé")


# Q4
def ville(soup):
    try:
        ville_tag=soup.find("h2", class_="mt-0")
        ville_text=ville_tag.text.strip()
        dernier_index=ville_text.rfind(", ")
        ville=ville_text[dernier_index+2:]
        return ville
    except:
        raise NonValide("Ville non trouvée.")

# Q5
def caracteristiques(soup):
    caracteristiques_balise = soup.find('p', class_='ad-section-title')
    if not caracteristiques_balise:
        raise NonValide("Caractéristiques non trouvées")
    return caracteristiques_balise.find_next("div")  

def type(soup):
    caracteristique = caracteristiques(soup)
    try:
        type_tag=caracteristique.find("span",string="Type")
        type=type_tag.find_next("span").text.strip()
        if type not in ["Maison","Appartement"]:
            raise NonValide("Type non valide.")
        return type
    except:
        raise NonValide("Type non trouvé.")

def surface(soup):
    caracteristique = caracteristiques(soup)
    surface_tag=caracteristique.find("span", string="Surface")
    if surface_tag:
        surface=surface_tag.find_next("span").text.replace("m²", "").strip()
        return surface
    return "-"

def nbrpieces(soup):
    caracteristique = caracteristiques(soup)
    nbrpieces_tag=caracteristique.find("span",string="Nb. de pièces")
    if nbrpieces_tag:
        nbrpieces=nbrpieces_tag.find_next("span").text.strip()
        return nbrpieces
    return "-"

def nbrchambres(soup):
    caracteristique = caracteristiques(soup)
    nbrchambres_tag=caracteristique.find("span",string="Nb. de chambres")
    if nbrchambres_tag:
        nbrchambres=nbrchambres_tag.find_next("span").text.strip()
        return nbrchambres
    return "-"

def nbrsdb(soup):
    caracteristique = caracteristiques(soup)
    nbrsdb_tag=caracteristique.find("span",string="Nb. de sales de bains")
    if nbrsdb_tag:
        nbrsdb=nbrsdb_tag.find_next("span").text.strip()
        return nbrsdb
    return "-"

def dpe(soup):
    caracteristique = caracteristiques(soup)
    dpe_tag = caracteristique.find("span", string="Consommation d'énergie (DPE)")
    if dpe_tag:
        dpe = dpe_tag.find_next("span").text.strip()
        if '(' in dpe:
            dpe = dpe.split('(')[0].strip()
        return dpe
    return "-"


# Q6
def informations(soup):
    try:
        ville_str = ville(soup)
        type_str = type(soup)
        surface_str = surface(soup)
        nbrpieces_str = nbrpieces(soup)
        nbrchambres_str = nbrchambres(soup)
        nbrsdb_str = nbrsdb(soup)
        dpe_str = dpe(soup)
        prix_str = prix(soup)

        return f"{ville_str},{type_str},{surface_str},{nbrpieces_str},{nbrchambres_str},{nbrsdb_str},{dpe_str},{prix_str}"
    except NonValide as e:
        raise NonValide(f"{e}")


# Q7
def get_max_page(soup):
    pagination = soup.select("ul.pagination li a")
    if pagination:
        nbDePages = [int(link.text) for link in pagination if link.text.isdigit()]
        return max(nbDePages) if nbDePages else 1
    return 1

def annonces_CSV():
    url = "https://www.immo-entre-particuliers.com/annonces/france-ile-de-france/vente"
    soup = getsoup(url)
    max_page = get_max_page(soup) 
    print(f"Total de pages : {max_page}")

    annonces_data = []
    annonces_urls = set()

    for page in range(1, max_page + 1): 
        page_url = f"https://www.immo-entre-particuliers.com/annonces/france-ile-de-france/vente/{page}"
        print(f"Page : {page}")
        soup = getsoup(page_url)

        annonces = soup.select('a[href^="/annonce-"]')
        
        for annonce in annonces:
            annonce_url = annonce['href']
            if not annonce_url.startswith("http"):
                annonce_url = "https://www.immo-entre-particuliers.com" + annonce_url
            
            if annonce_url in annonces_urls:
                continue  
            annonces_urls.add(annonce_url)

            try:
                annonce_soup = getsoup(annonce_url)  
                info = informations(annonce_soup)
                if info:
                    annonces_data.append(info)
                    print(f"{info}")
            except NonValide as e:
                print(f"Annonce invalide {annonce_url} : {e}")
                continue  

    if annonces_data:
        with open("annonces.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Ville", "Type", "Surface", "NbrPieces", "NbrChambres", "NbrSdb", "DPE", "Prix"])
            for row in annonces_data:
                writer.writerow(row.split(','))

        print("Fini")

annonces_CSV() 

#################################################################
#             Deuxième jalon : Nettoyage des données            #
#################################################################


# Q8
annonces = pd.read_csv("annonces.csv")


##      Valeurs manquantes

# Q9
annonces["DPE"] = annonces["DPE"].replace("-", "Vierge")

# Q10
cols = ["Surface", "NbrPieces", "NbrChambres", "NbrSdb"]

annonces[cols] = annonces[cols].replace("-", np.nan)
annonces[cols] = annonces[cols].astype(float)
moyenne = annonces[cols].mean()
annonces[cols] = annonces[cols].fillna(moyenne[cols])

annonces.dropna(inplace=True)


##      Colonnes “Type" et “DPE”

# Q11
annonces = pd.get_dummies(annonces, columns=["Type", "DPE"], dtype='int')


##      Colonne “Ville”

# Q12
villes = pd.read_csv("cities.csv")

# Q13
annonces["Ville"] = (
    annonces["Ville"]
    .str.lower()  
    .str.replace("-", " ", regex=True) 
    .str.replace("'", "", regex=True)  
    .str.replace(r"[éèê]", "e", regex=True) 
    .str.replace(r"[ô]", "o", regex=True)  
    .str.replace(r"[î]", "i", regex=True)  
    .str.replace(r"[û]", "u", regex=True)  
    .str.replace(r"[à]", "a", regex=True)  
)

villes["label"] = (
    villes["label"]
    .str.lower()
    .str.replace("-", " ", regex=True)
    .str.replace("'", "", regex=True)
    .str.replace(r"[éèê]", "e", regex=True)
    .str.replace(r"[ô]", "o", regex=True)
    .str.replace(r"[î]", "i", regex=True)
    .str.replace(r"[û]", "u", regex=True)
    .str.replace(r"[à]", "a", regex=True)
)


# Q14
annonces = annonces.merge(villes[['label', 'latitude', 'longitude']], left_on="Ville", right_on="label")

annonces.drop(columns=["Ville", "label"], inplace=True)


#   TEST
villes.to_csv("cities.csv", index=False)

annonces.to_csv("annonces.csv", index=False)

print(annonces.to_string())
print(annonces.info())
print(moyenne)



#################################################################
#               Troisième jalon : Apprentissage                 #
#################################################################


##      Préparation de données d’apprentissage et de test

# Q15
X=annonces.drop(columns=["Prix"])
y=annonces["Prix"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=49)


##      Premier modèle : Régression Linéaire

# Q16
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
print("Régression linéaire R2_score:", r2_lr)

# Q17
lr_norm = make_pipeline(MinMaxScaler(), LinearRegression())
lr_norm.fit(X_train, y_train)
y_pred_lr_norm = lr_norm.predict(X_test)
r2_lr_norm = r2_score(y_test, y_pred_lr_norm)
print("Normalisation LR:", r2_lr_norm)

lr_std = make_pipeline(StandardScaler(), LinearRegression())
lr_std.fit(X_train, y_train)
y_pred_lr_std = lr_std.predict(X_test)
r2_lr_std = r2_score(y_test, y_pred_lr_std)
print("Standardisation LR:", r2_lr_std)

# Q18
print("\nMéthode\t\t\t\tR² Score")
print("-" * 40)
print(f"LR\t\t\t\t{r2_lr:.4f}")
print(f"Normalisation + LR\t\t{r2_lr_norm:.4f}")
print(f"Standardisation + LR\t\t{r2_lr_std:.4f}\n")

# Il n'y a pas d'amélioration des prédictions due au pré-traitement pour ce jeu de données et cette méthode.

##      Deuxième modèle : Arbre de Décision

# Q19
for depth in range (4,10):
    ad_depth=DecisionTreeRegressor(max_depth=depth, random_state=49)
    ad_depth.fit(X_train, y_train)
    y_pred_ad_depth=ad_depth.predict(X_test)
    r2_ad_depth=r2_score(y_test, y_pred_ad_depth)
    print(f"R^2 Score (Decision Tree, max_depth={depth}): {r2_ad_depth}")

ad=DecisionTreeRegressor(max_depth=7)
ad.fit(X_train, y_train)
y_pred_ad=ad.predict(X_test)
r2_ad=r2_score(y_test, y_pred_ad)

ad_norm=make_pipeline(MinMaxScaler(), DecisionTreeRegressor(max_depth=7))
ad_norm.fit(X_train, y_train)
y_pred_ad_norm=ad_norm.predict(X_test)
r2_ad_norm=r2_score(y_test, y_pred_ad_norm)

ad_std=make_pipeline(StandardScaler(), DecisionTreeRegressor(max_depth=7))
ad_std.fit(X_train, y_train)
y_pred_ad_std=ad_std.predict(X_test)
r2_ad_std=r2_score(y_test, y_pred_ad_std)

# L’augmentation de max_depth améliore les résultats, jusqu’à max_depth=7, après quoi la performance diminue.
# Le pré-traitement améliore légèrement les prédictions.
# Les scores obtenus sont satisfaisants.
# On va conserver max_depth=7.

# Q20
print("\nMéthode\t\t\t\tR² Score")
print("-" * 40)
print(f"AD\t\t{r2_ad:.4f}")
print(f"Normalisation + AD\t\t{r2_ad_norm:.4f}")
print(f"Standardisation + AD\t\t{r2_ad_std:.4f}\n")


##      Troisième modèle : N plus proches voisins

# Q21
knn = KNeighborsRegressor(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
r2_knn = r2_score(y_test, y_pred_knn)
print("KNN (n=4) sans prétraitement:", r2_knn)

knn_norm = make_pipeline(MinMaxScaler(), KNeighborsRegressor(n_neighbors=4))
knn_norm.fit(X_train, y_train)
y_pred_knn_norm = knn_norm.predict(X_test)
r2_knn_norm = r2_score(y_test, y_pred_knn_norm)
print("KNN + Normalisation:", r2_knn_norm)

knn_std = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=4))
knn_std.fit(X_train, y_train)
y_pred_knn_std = knn_std.predict(X_test)
r2_knn_std = r2_score(y_test, y_pred_knn_std)
print("KNN + Standardisation:", r2_knn_std)

# Standardisation + KNN avec n=5
knn_std_5 = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
knn_std_5.fit(X_train, y_train)
y_pred_knn_std5 = knn_std_5.predict(X_test)
r2_knn_std_5 = r2_score(y_test, y_pred_knn_std5)
print("KNN + Standardisation (n=5):", r2_knn_std_5)
# La méthode des K plus proches voisins (KNN) offre les meilleurs résultats avec un score R2 de 0.3708. Passer de n_neighbors=4 à n_neighbors=5 n’a pas significativement amélioré les performances, donc garder n=4.

# Q22
print("\nMéthode\t\t\t\tR² Score")
print("-" * 40)
print(f"KNN\t\t\t\t{r2_knn:.4f}")
print(f"Normalisation + KNN\t\t{r2_knn_norm:.4f}")
print(f"Standardisation + KNN\t\t{r2_knn_std:.4f}\n")


##      Discussions sur le jeu de données

# Q23
print("\nMéthode\t\t\t\tR² Score")
print("-" * 40)
print(f"LR\t\t\t\t{r2_lr:.4f}")
print(f"AD\t\t\t\t{r2_ad_norm:.4f}")
print(f"KNN\t\t\t\t{r2_knn_norm:.4f}\n")

# On note la méthode M = Normalisation + AD

##      Visualisation des résultats de test

# Q24
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linestyle='--', linewidth=1)
plt.scatter(y_test, y_pred_ad_norm, color='green', marker='*', s=100) 
plt.title("y_test versus estimation (Arbre de Décision Normalisé)")
plt.xlabel("y_test")
plt.ylabel("estimation")
plt.show()


##      Réduction du nombre d’attributs

# Q25
pca=PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)
print(f"Ratio de variance expliquée : {pca.explained_variance_ratio_}")
print(f"Somme du ratio de variance expliquée : {pca.explained_variance_ratio_.sum()}")

# Q26
ad_norm_pca=make_pipeline(MinMaxScaler(), DecisionTreeRegressor(max_depth=7))
ad_norm_pca.fit(X_train_pca, y_train)
y_pred_pca=ad_norm_pca.predict(X_test_pca)
r2_pca=r2_score(y_test, y_pred_pca)
print("Score R2 après réduction (PCA) :", r2_pca)


##      Matrice de corrélation

# Q27
corr_matrix=annonces.corr()

sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.title("Matrice de Corrélation")
plt.show()

# Q28
corr_prix=corr_matrix["Prix"].drop("Prix")
max_corr_attr=corr_prix.idxmax()
min_corr_attr=corr_prix.idxmin() 

print(f"L'attribut le plus corrélé positivement avec le prix : {max_corr_attr} ({corr_prix[max_corr_attr]})")
print(f"L'attribut le plus corrélé négativement avec le prix : {min_corr_attr} ({corr_prix[min_corr_attr]})")

# Q29

top5_attributs=corr_matrix["Prix"].sort_values(ascending=False).index[1:6] 

annonces_top5_attributs=annonces[top5_attributs.tolist() + ["Prix"]]

X_reduit=annonces_top5_attributs.drop(columns=["Prix"])
y_reduit=annonces_top5_attributs["Prix"]

X_train_red, X_test_red, y_train_red, y_test_red=train_test_split(X_reduit, y_reduit, test_size=0.25, random_state=49)

ad_norm_red=make_pipeline(MinMaxScaler(), DecisionTreeRegressor(max_depth=7))
ad_norm_red.fit(X_train_red, y_train_red)
y_pred_red=ad_norm_red.predict(X_test_red)
r2_reduit=r2_score(y_test_red, y_pred_red)

print("Score R2 après réduction du nombre d'attributs :",r2_reduit)

# On constate que le score R² est inférieur aux scores obtenus précédemment avec l'ensemble des attributs.
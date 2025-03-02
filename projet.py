from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
import numpy as np

#   Q1
def getsoup(url):
    return BeautifulSoup(requests.get(url).text,'html.parser')


#   Q2
class NonValide(Exception):
    pass


#   Q3
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


#   Q4
def ville(soup):
    try:
        ville_tag=soup.find("h2", class_="mt-0")
        ville_text=ville_tag.text.strip()
        dernier_index=ville_text.rfind(", ")
        ville=ville_text[dernier_index+2:]
        return ville
    except:
        raise NonValide("Ville non trouvée.")

#   Q5
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


#   Q6
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


#   Q7
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

#   Q8
annonces = pd.read_csv("annonces.csv")

#   Q9
annonces["DPE"] = annonces["DPE"].replace("-", "Vierge")

#   Q10
cols = ["Surface", "NbrPieces", "NbrChambres", "NbrSdb"]

annonces[cols] = annonces[cols].replace("-", np.nan)
annonces[cols] = annonces[cols].astype(float)
moyenne = annonces[cols].mean()
annonces[cols] = annonces[cols].fillna(moyenne[cols])

annonces.dropna(inplace=True)

#   Q11
annonces = pd.get_dummies(annonces, columns=["Type", "DPE"], dtype='int')


#   Q12
villes = pd.read_csv("cities.csv")

#   Q13
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


#   Q14
annonces = annonces.merge(villes[['label', 'latitude', 'longitude']], left_on="Ville", right_on="label")

annonces.drop(columns=["Ville", "label"], inplace=True)


#   test
villes.to_csv("cities.csv", index=False)

annonces.to_csv("annonces.csv", index=False)

print(annonces.to_string())
print(annonces.info())
print(moyenne)


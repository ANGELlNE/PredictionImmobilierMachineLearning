import requests
from bs4 import BeautifulSoup

# Q1
def getsoup(url):
    return BeautifulSoup(requests.get(url).text,'html.parser')

# Q2 
class NonValide(Exception):
    pass

# Q3
def prix(soup):
    prix_tag=soup.find("p", class_="product-price")
    if not prix_tag:
        raise NonValide("Prix non trouvé")
    prix_text=prix_tag.text.replace("€", "").replace(" ", "").strip()
    prix=int(prix_text)
    if prix<10000:
        raise NonValide("Prix trop bas.")
    return prix

# Q4
def ville(soup):
    ville_tag=soup.find("h2", class_="mt-0")
    if not ville_tag:
        raise NonValide("Ville non trouvée.")
    ville_text=ville_tag.text.strip()
    dernier_index=ville_text.rfind(", ")
    ville=ville_text[dernier_index+2:]
    return ville

# Q5

def type(soup):
    type_tag=soup.find("span",string="Type")
    if type_tag:
        type=type_tag.find_next("span").text.strip()
        if type not in ["Maison","Appartement"]:
            raise NonValide("Type non valide.")
        return type
    return "-"

def surface(soup):
    surface_tag=soup.find("span", string="Surface")
    if surface_tag:
        surface=surface_tag.find_next("span").text.strip()
        return surface
    return "-"

def nbrpieces(soup):
    nbrpieces_tag=soup.find("span",string="Nb. de pièces")
    if nbrpieces_tag:
        nbrpieces=nbrpieces_tag.find_next("span").text.strip()
        return nbrpieces
    return "-"

def nbrchambres(soup):
    nbrchambres_tag=soup.find("span",string="Nb. de chambres")
    if nbrchambres_tag:
        nbrchambres=nbrchambres_tag.find_next("span").text.strip()
        return nbrchambres
    return "-"

def nbrsdb(soup):
    nbrsdb_tag=soup.find("span",string="Nb. de sales de bains")
    if nbrsdb_tag:
        nbrsdb=nbrsdb_tag.find_next("span").text.strip()
        return nbrsdb
    return "-"

def dpe(soup):
    dpe_tag=soup.find("span",string="Consommation d'énergie (DPE)")
    if dpe_tag:
        dpe=dpe_tag.find_next("span").text.strip()
        return dpe
    return "-"

# TEST
url = "https://www.immo-entre-particuliers.com/annonce-landes-dax/409314-appartement-de-caractere"

soup=getsoup(url)

print("Prix:",prix(soup))
print("Ville:",ville(soup))
print("Type:",type(soup))
print("Surface:",surface(soup))
print("Nombre de pièces:",nbrpieces(soup))
print("Nombre de chambres:",nbrchambres(soup))
print("Nombre de salles de bains:",nbrsdb(soup))
print("Consommation d'énergie (DPE):",dpe(soup))

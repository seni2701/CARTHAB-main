"""
SERIF-TOOLBOX - Étape 2
2.5.1 Création des transects polygonnaux

Development info :

Python version : 3.11.5
virtual environment : conda
Python IDE : PyCharm
numpy : 1.26.0
geopandas version : 0.9.0
shapely version : 2.0.2
pandas version : 2.1.4
geojson version : 3.1.0

Code info : Ce code Python crée les transects polygonnaux pour la partie surfacique à partir d'une ligne de rivière,
puis les découpe en utilisant des zones actives et des zones d'eau comme masques.

1. Chargement et préparation des données :

Définition des chemins d'accès aux fichiers (ligne de rivière, zones actives, zones d'eau) et du seuil de filtrage
pour les petits polygones. Lecture des fichiers (ligne de rivière, zones actives, zones d'eau) avec geopandas.
Détermination du système de coordonnées de référence (CRS) de la rivière. Reprojection des données dans le CRS de la
rivière si nécessaire.

2. Création de transects bruts :

Création de points le long de la ligne de rivière à intervalles réguliers.
Génération de polygones de Voronoi à partir de ces points. Ces polygones représentent les transects bruts.
Sauvegarde des transects bruts dans un fichier shapefile.

3. Découpage des transects :

Découpage des transects bruts en utilisant les zones actives et les zones d'eau comme masques. Cela permet de créer
deux ensembles de transects : ceux qui se trouvent dans les zones actives et ceux qui se trouvent dans les zones d'eau.

4. Nettoyage et filtrage :

Explosion des multi-polygones en polygones simples.
Attribution d'un identifiant unique à chaque transect.
Calcul de la surface de chaque transect.
Filtrage des transects pour ne conserver que ceux dont la surface est supérieure au seuil défini.

5. Enregistrement des résultats :

Conversion des types de données des colonnes.
Sauvegarde des transects découpés et filtrés (zones actives et zones d'eau) dans des fichiers shapefiles distincts.

6. Journalisation :

Enregistrement des informations d'exécution (date, heure, durée) dans un fichier journal.

Auteur : Mathias Chabal - INRS
"""


# Importation des packages nécessaires
import geopandas as gpd
from pathlib import Path
from neptune.tools import crs_river, get_col_dtype_sf, get_columns_sf
from neptune.geometry import points_along_line, voronoi_diagrams
from neptune.logger import Logger
from warnings import filterwarnings

### Chemins d'accès à modifier ###
line_path = Path(r'D:\CPD\Ligne_BA_CPD.shp') # Ligne centrale de la bande active
active_path = Path(r'D:\CPD\Active_CPD.shp') # Masque de la bande active
water_path = Path(r'D:\CPD\Eau_CPD.shp') # Masque de l'eau
out_raw_path = Path(r'D:\CPD\Transects_brut_CPD.shp') # Chemin vers le fichier de sortie des transects bruts
out_active_path = Path(r'D:\CPD\Transects_CPD.shp') # Chemin vers le fichier de sortie des transects 'active'
out_water_path = Path(r'D:\CPD\Transects_Eau_CPD.shp') # Chemin vers le fichier de sortie des transects 'eau'
rhs_path = Path(r'D:\CPD\rhs_CPD.shp') # Chemin vers le fichier RHS de la rivière
filtering_threshold = 4  # Garde les polygones qui sont plus que la valeur (m2)

### Début du code ###

if not rhs_path.exists():
    raise FileNotFoundError(f"No such file found for the rhs : {rhs_path}")

# Utilisation de la fonction time pour le suivi de la durée d'exécution du script
filterwarnings('ignore')  #  Ignorer tous les avertissements
river_code = line_path.parent.name  # Obtention du nom de la rivière
log = Logger(river_code, line_path.parent)
crs = crs_river(river_code)  # Obtention du système de coordonnées de référence (CRS) de la rivière

print(f'Treating river : {river_code}')  # Affiche la rivière en cours de traitement

line = gpd.read_file(line_path, engine="pyogrio", use_arrow=True)  # Lire le fichier de la ligne de la rivière

# Vérification et mise à jour du CRS de la ligne, si nécessaire
if line.crs.to_epsg() != crs:
    line.to_crs(crs, inplace=True)

print('Creating points along the line')
# Création de points le long de la ligne de la rivière
points_for_vor = points_along_line(line, 5)

print('Creating voronoï polygons')
# Création de polygones de Voronoï à partir des points créés précédemment
vor_polys = voronoi_diagrams(points_for_vor, return_list=False)

# Sauvegarde des polygones dans un fichier
vor_polys.to_file(out_raw_path, engine="pyogrio", use_arrow=True)

# Lecture des fichiers active et water
active = gpd.read_file(active_path, engine="pyogrio", use_arrow=True)
water = gpd.read_file(water_path, engine="pyogrio", use_arrow=True)

# Vérification et mise à jour du CRS de 'active' et 'water', si nécessaire
if active.crs.to_epsg() != crs:
    active.to_crs(crs, inplace=True)
if water.crs.to_epsg() != crs:
    water.to_crs(crs, inplace=True)

print('Cliping polygons with the active and water mask')
# Découpage des polygones avec le masque 'active' et 'water'
transects_active = vor_polys.copy()
transects_active['geometry'] = vor_polys.intersection(active.union_all())
transects_water = vor_polys.copy()
transects_water['geometry'] = vor_polys.intersection(water.union_all())

print('Exploding multi polygons')
# Explose les polygones multiples en polygones simples
transects_active = transects_active.explode()
transects_water = transects_water.explode()

# Renommez la colonne 'ID' en 'ORI_ID' et créez une nouvelle colonne 'ID' avec une plage de numéros
transects_active = transects_active.rename(columns={'ID': 'ORI_ID'})
transects_water = transects_water.rename(columns={'ID': 'ORI_ID'})
transects_water['ID'] = range(len(transects_water))
transects_active['ID'] = range(len(transects_active))

print('Filtering small polygons')
# Ajoute une nouvelle colonne 'Surface' avec l'aire de chaque polygone.
# Supprime ensuite les polygones dont la surface est inférieure au seuil de filtrage
transects_active['Surface'] = transects_active.area
transects_water['Surface'] = transects_water.area
transects_active = transects_active[transects_active['Surface'] > filtering_threshold]
transects_water = transects_water[transects_water['Surface'] > filtering_threshold]

rhs = gpd.read_file(rhs_path, engine="pyogrio", use_arrow=True).to_crs(crs)
rhs_poly = rhs[rhs["TYPECE"] == 21].union_all()

print('Assigning the variable Lac to the transects')
water_lake_mask = transects_water.intersects(rhs_poly)
transects_water["Lac"] = water_lake_mask.map({True: 1, False: 0})

active_lake_mask = transects_active.intersects(rhs_poly)
transects_active["Lac"] = active_lake_mask.map({True: 1, False: 0})

# Convertit les ensembles de données pour qu'ils aient le bon type de données pour chaque colonne
active_cols = transects_active.columns
transects_active = transects_active[get_columns_sf(active_cols)].astype(get_col_dtype_sf(active_cols), errors='ignore')
water_cols = transects_water.columns
transects_water = transects_water[get_columns_sf(water_cols)].astype(get_col_dtype_sf(water_cols), errors='ignore')

transects_active.to_file(out_active_path, engine="pyogrio", use_arrow=True)
transects_water.to_file(out_water_path, engine="pyogrio", use_arrow=True)

log.stop()
log.print_log()

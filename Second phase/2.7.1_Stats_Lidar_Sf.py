"""
SERIF-TOOLBOX - Étape 2
2.7.1 Statistique lidar brut (surfacique)

Development info :

Python version : 3.11.5
virtual environment : conda
Python IDE : PyCharm
numpy : 1.26.0
geopandas version : 0.9.0
shapely version : 2.0.2
pandas version : 2.1.4
PDAL version : 3.2.3
laspy version : 2.5.1
json version : 2.0.9
geojson version : 3.1.0
tqdm version : 4.66.1

Code info : Ce code Python traite des données de transects polygonnaux et des données LIDAR pour calculer le 25e
percentile (LB_Q25) de l'altitude pour chaque transect.

1. Chargement et préparation des données :

Définition des chemins vers les données (transects, index LIDAR) et du répertoire de travail.
Lecture des données de transects et de l'index LIDAR.
Reprojection des transects si nécessaire.
Tri des transects par PK décroissant.

2. Association des fichiers LIDAR aux transects :

Pour chaque transect de type "Surface", le code identifie les fichiers LIDAR qui l'intersectent en utilisant
l'index LIDAR.
Les noms des fichiers LIDAR intersectant chaque transect sont stockés dans une nouvelle colonne 'File' du
GeoDataFrame des transects, séparés par des virgules si plusieurs fichiers intersectent le même transect.

3. Traitement par fichier LIDAR :

Le code itère ensuite sur chaque combinaison unique de fichier LIDAR.
Pour chaque fichier LIDAR, il sélectionne le sous-ensemble de transects qui lui sont associés.

Les données LIDAR de tous les fichiers associés à un transect sont combinées en un seul tableau NumPy. Si les
fichiers LIDAR ont un CRS différent de celui des transects, les coordonnées sont transformées.

4. Traitement par transect (pour chaque fichier LIDAR) :

Pour chaque transect du sous-ensemble courant, le code effectue les opérations suivantes :
Création d'une zone tampon autour du centroïde du polygone du transect, dont la taille est 60% de la largeur
du transect (WAT_WIDTH). Découpage entre le polygone original du transect et le buffer. Cela permet de se concentrer
sur la zone centrale du transect, plus susceptible de représenter la surface de l'eau et non pas un arbre ou la berge.
Création d'un objet matplotlib.path.Path à partir des coordonnées du polygone découpé.
Identification des points LIDAR à l'intérieur du polygone découpé.
Calcul du 25e percentile de l'altitude (coordonnée Z) des points LIDAR sélectionnés.
Stockage de la valeur (LB_Q25) dans le GeoDataFrame des transects.

5. Enregistrement des résultats :

Sélection des colonnes nécessaires et conversion des types de données.
Enregistrement du GeoDataFrame mis à jour (avec la colonne LB_Q25) dans un fichier shapefile.

6. Journalisation :

Enregistrement des informations d'exécution (date, heure, durée) dans un fichier journal.


Auteur : Mathias Chabal - INRS
"""

import geopandas as gpd
import laspy as lp
from tqdm import tqdm
from shapely import contains_xy
from warnings import filterwarnings
import numpy as np
from pathlib import Path
from neptune.tools import crs_river, get_columns_sf, get_col_dtype_sf
from neptune.geometry import drop_z
from neptune.logger import Logger


### Paramétres à modifier ###
river_directory = Path(r'D:\CPD') # Dossier de travail
transects_path = Path(r'D:\CPD\Transects_Eau_CPD.shp') # Transects du masque eau
out_file = Path(r'D:\CPD\Transects_LB_CPD.shp') # Fichier en sortie

river_code = transects_path.parent.name
log = Logger(river_code, river_directory)
CRS = crs_river(river_code)

### Début du code ###
print(f'Treating river : {river_code}')
filterwarnings("ignore")

transects = gpd.read_file(transects_path)

if transects.crs.to_epsg() != CRS:
    transects.to_crs(CRS, inplace=True)

if transects.geometry.has_z.any():
    transects['geometry'] = transects['geometry'].apply(drop_z)


transects = transects.sort_values(by='PK', ascending=False)  # Tri des transects par PK décroissant
transects['WAT_WIDTH'] = transects['geometry'].minimum_bounding_radius()*2

ori_geoms = transects["geometry"].copy(deep=True)
centros = ori_geoms.centroid.buffer((transects['WAT_WIDTH'] * 0.6) / 2)
transects["geometry"] = ori_geoms.intersection(centros)

for block_fp, block in transects.groupby("block_id"):

    ld = lp.read(block_fp)
    lidar_data = ld.xyz
    classification = np.array(ld.classification).reshape(-1, 1)

    for i, poly in tqdm(block["geometry"].items(), total=len(block), desc=f'Processing block: {Path(block_fp).name}'):

        if not poly.is_valid:
            print(f'Invalid geometry for transect {i}')
            continue  # Passe au transect suivant si le polygone n'est pas valide

        minx, miny, maxx, maxy = poly.bounds
        bbox_mask = ((lidar_data[:, 0] >= minx) & (lidar_data[:, 0] <= maxx) &
                     (lidar_data[:, 1] >= miny) & (lidar_data[:, 1] <= maxy))

        clipped_lidar = lidar_data[bbox_mask]
        clipped_classification = classification[bbox_mask]

        mask = contains_xy(poly, clipped_lidar[:, 0], clipped_lidar[:, 1]) # garde les dimensions x et y

        if np.sum(mask) > 3: # nécessite un minimum de 4 points

            classification_mask = (clipped_classification == 9).flatten() & mask

            if np.sum(classification_mask) > 3:
                masked_lidar_data = clipped_lidar[classification_mask]

            else:
                masked_lidar_data = clipped_lidar[mask]  # Sélection des points lidar à l'intérieur du polygone

            q25_val = np.percentile(masked_lidar_data[:, 2], 25, axis=0)  # Calcul du quantile 25 des altitudes sur la 3e
            # dimension du tableau (correspond aux valeurs de Z)
            transects.loc[i, 'LB_Q25'] = q25_val

transects["geometry"] = ori_geoms

columns = transects.columns
# Garde seulement les colonnes nécessaires et modifie le type si mauvais
transects = transects[get_columns_sf(columns)].astype(get_col_dtype_sf(columns), errors='ignore')
transects.to_file(out_file, index=False)

print(f'Saved as : {str(river_directory.joinpath(out_file))}')

log.stop()
log.print_log()

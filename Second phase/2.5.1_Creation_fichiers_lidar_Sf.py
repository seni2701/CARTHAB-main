"""
SERIF-TOOLBOX - Étape 2
2.4 Création des fichiers lidar

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

Code info :

Ce code Python fusionne des fichiers LIDAR en utilisant PDAL (Point Data Abstraction Library) et un pipeline JSON.

1. Chargement et préparation des données :

Définition des chemins vers les données (ligne centrale, index LIDAR) et les fichiers de pipeline PDAL.
Lecture de la ligne centrale et de l'index LIDAR avec geopandas.
Détermination du système de coordonnées de référence (CRS) de la rivière.
Reprojection des données dans le CRS de la rivière si nécessaire.
Création du répertoire de sortie pour les fichiers LIDAR fusionnés.

2. Traitement par tronçon de rivière :

Itération sur chaque tronçon de la ligne centrale.
Création d'une zone tampon autour du tronçon.
Sélection des fichiers LIDAR de l'index qui intersectent la zone tampon.

3. Fusion des fichiers LIDAR :

Itération sur les fichiers LIDAR sélectionnés pour chaque tronçon.
Utilisation de PDAL et d'un pipeline JSON pour fusionner les fichiers LIDAR.
Le code utilise différents fichiers de pipeline JSON en fonction du CRS des fichiers LIDAR et du fait qu'il s'agisse
du premier fichier à fusionner ou non. Cela permet de gérer les transformations de CRS et d'optimiser le processus de
fusion.
Le pipeline PDAL est configuré avec les chemins de fichiers, le CRS et la zone tampon.
Exécution du pipeline PDAL pour fusionner les fichiers LIDAR.
Gestion des erreurs potentielles lors de la fusion des fichiers LIDAR.

4. Enregistrement des résultats :

Les fichiers LIDAR fusionnés sont enregistrés dans le répertoire de sortie.

5. Journalisation :

Enregistrement des informations d'exécution (date, heure, durée) dans un fichier journal.

Auteur : Mathias Chabal - INRS
"""

import geopandas as gpd
from tqdm import tqdm
from pdal import Pipeline
import json
from pathlib import Path
from neptune.tools import crs_river, calculate_blocks, split_df
from neptune.logger import Logger

### Paramètres à modifier ###
river_directory = Path(r'D:\CPD')   # Dossier de travail
index_path = Path(r'D:\CPD\Lidar\Index\Index_LB_CPD.shp') # Chemin de l'index LIDAR
transects_path = Path(r'D:\CPD\Transects_Eau_CPD.shp') # Transects de l'eau
merged_lidar_dir = Path(r"D:\CPD\Lidar\Merged") # dossier où enregistrer les fichiers LIDAR

river_code = river_directory.name
log = Logger(river_code, river_directory)
CRS = crs_river(river_code)

transects = gpd.read_file(transects_path, engine="pyogrio", use_arrow=True)
lidar_index = gpd.read_file(index_path, engine="pyogrio", use_arrow=True)

if transects.crs.to_epsg() != CRS:
    transects.to_crs(CRS, inplace=True)

if lidar_index.crs.to_epsg() != CRS:
    lidar_index.to_crs(CRS, inplace=True)

transects.sort_values("PK", inplace=True)
sindex = lidar_index.sindex

OPTIMAL_SEGMENT_LENGTH = 10 # en km
length = transects["PK"].max() / 1_000 # passer en km
n_blocks = calculate_blocks(length, OPTIMAL_SEGMENT_LENGTH)
blocks = split_df(transects, n_blocks)

merged_lidar_dir.mkdir(exist_ok=True)

for i, block in tqdm(enumerate(blocks), total=len(blocks)):

    block_fp = merged_lidar_dir.joinpath(river_code + f"_block_{i}.laz")
    transects.loc[block.index, "block_id"] = str(block_fp)

    block_geom = block.union_all()
    indices = sindex.query(block_geom, predicate="intersects")
    files = lidar_index.loc[indices, "Chemin"].to_list()

    pipeline = {
        'pipeline': files + [
            {
                "type": "filters.merge"
            },
            {
                "type": "filters.crop",
                "polygon": block_geom.wkt
            },
            {
                "type": "writers.las",
                "filename": str(block_fp)
            }
        ]
    }

    pipline = Pipeline(json.dumps(pipeline))
    pipline.execute()

transects.to_file(transects_path, engine="pyogrio", use_arrow=True)

log.stop()
log.print_log()

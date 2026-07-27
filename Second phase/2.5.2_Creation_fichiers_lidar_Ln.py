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

import warnings
import geopandas as gpd
from tqdm import tqdm
from pdal import Pipeline
import json
from pathlib import Path
from neptune.tools import crs_river
from neptune.logger import Logger


### Paramètres à modifier ###
# Création des chemins d'accès
river_directory = Path(r'D:\CPD')  # Dossier de travail
centerline_path = Path(r'D:\CPD\Ligne_Pr_Tr_CPD.shp') # Ligne centrale des tributaires et principale
index_path = Path(r'D:\CPD\Lidar\Index\Index_LB_CPD.shp') # Chemin de l'index LIDAR

### Début du code ###

# Obtention du nom de la rivière
river_code = river_directory.name
log = Logger(river_code, river_directory)
# Obtention du système de coordonnées de référence (CRS) de la rivière
CRS = crs_river(river_code)

# Lecture de la ligne centrale
centerline = gpd.read_file(centerline_path, engine="pyogrio", use_arrow=True)

# Vérification et mise à jour du CRS de la ligne centrale, si nécessaire
if centerline.crs.to_epsg() != CRS:
    centerline.to_crs(CRS, inplace=True)

# Lecture de l'index du lidar
lidar_index = gpd.read_file(index_path, engine="pyogrio", use_arrow=True)
lidar_index["ORI_CRS"] = lidar_index["ORI_CRS"].astype(int)
lidar_index["Chemin"] = lidar_index["Chemin"].str.strip()
sindex = lidar_index.sindex

if not lidar_index["Chemin"].apply(lambda p: Path(p).exists()).all():
    raise FileNotFoundError("Some files in the lidar index do not exist.")

# Vérification et mise à jour du CRS de l'index du lidar, si nécessaire
if not lidar_index.crs.to_epsg() == CRS:
    lidar_index.to_crs(CRS, inplace=True)

out_file_dir = river_directory.joinpath('Lidar/Merged')
out_file_dir.mkdir(exist_ok=True)

centerline = centerline[centerline['Methode'] == 'Lineaire']

warnings = []
# Boucle qui passe par chaque tronçon de la ligne centrale
for index, row in tqdm(centerline.iterrows(), total=centerline.shape[0]):

    geom = row.geometry.buffer(15)

    indices = sindex.query(geom, predicate="intersects")
    sub_gdf = lidar_index.loc[indices]

    if sub_gdf.empty:
        warnings.append(f"The tributary '{row.Nom}' does not intersect any LiDAR files.")
        continue

    if not sub_gdf.buffer(0.1).union_all().contains(row.geometry):
        warnings.append(f"The tributary '{row.Nom}' is not fully inside the LiDAR files.")
        continue

    crs_mask = sub_gdf["ORI_CRS"] != CRS

    if crs_mask.any():
        for_reprojection = sub_gdf[crs_mask]["Chemin"]
        ORI_CRS = sub_gdf[crs_mask]["ORI_CRS"].iloc[0]

        for file in for_reprojection:
            pipeline = {
                'pipeline': [
                    {
                        "type": "readers.las",
                        "filename": file
                    },
                    {
                        "type": "filters.reprojection",
                        "in_srs": f"EPSG:{ORI_CRS}",
                        "out_srs": f"EPSG:{CRS}"
                    },
                    {
                        "type": "writers.las",
                        "filename": file
                    }
                ]
            }

            pipline = Pipeline(json.dumps(pipeline))
            pipline.execute()

    files = sub_gdf["Chemin"].to_list()
    out_fp = out_file_dir.joinpath(f"Merged_{row.Nom}.laz")

    if out_fp.exists():
        print(f"File '{out_fp.name}' already exists.")
        continue

    temp_files = []

    for i, file in enumerate(files):
        temp_fp = out_file_dir.joinpath(f"temp_{row.Nom}_{i}.laz")
        temp_files.append(str(temp_fp))

        pipeline = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": file
                },
                {
                    "type": "filters.crop",
                    "polygon": geom.wkt
                },
                {
                    "type": "writers.las",
                    "filename": str(temp_fp),
                    "a_srs": f"EPSG:{CRS}"
                }
            ]
        }

        pipline = Pipeline(json.dumps(pipeline))
        pipline.execute()


    if len(temp_files) > 0:

        pipeline_merge = {
            "pipeline": temp_files + [
                {
                    "type": "filters.merge"
                },
                {
                    "type": "writers.las",
                    "filename": str(out_fp),
                    "a_srs": f"EPSG:{CRS}"
                }
            ]
        }

        pipline = Pipeline(json.dumps(pipeline_merge))
        n_points = pipline.execute()

        if n_points == 0:
            warnings.append(
                f"The tributary '{row.Nom}' has no points, number of files treated: {len(files)}"
            )

    for f in temp_files:
        Path(f).unlink(missing_ok=True)

if warnings:
    print("\n".join(warnings))

log.stop()
log.print_log()

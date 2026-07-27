"""
SERIF-TOOLBOX - Étape 2
2.3 Création de l'index lidar

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

Ce code Python crée un index spatial pour les fichiers LIDAR (LAS/LAZ) situés dans un dossier.

1. Initialisation et paramètres :

Définition du répertoire de travail (contenant les fichiers LIDAR) et du chemin de sortie pour l'index spatial.
Récupération du code de la rivière et du CRS associé.

2. Recherche des fichiers LIDAR :

Recherche de tous les fichiers avec les extensions .las ou .laz dans le répertoire LIDAR.

3. Création de l'index spatial :

Création d'un GeoDataFrame vide pour stocker les informations d'indexation. Les colonnes incluent l'ID,
le nom du fichier, le chemin du fichier, la densité de points, le CRS d'origine et la géométrie (emprise du fichier
LIDAR).
Itération sur chaque fichier LIDAR trouvé :
Lecture des métadonnées du fichier LIDAR avec laspy.
Récupération du CRS du fichier LIDAR.
Calcul de l'emprise du fichier LIDAR (bounding box) à partir des coordonnées min/max.
Transformation du CRS de l'emprise si le CRS du fichier LIDAR est différent du CRS de la rivière.
Calcul de la densité de points du fichier LIDAR.
Ajout des informations (nom du fichier, chemin, densité de points, CRS d'origine, emprise) à une nouvelle ligne
du GeoDataFrame.

4. Enregistrement de l'index spatial :

Attribution d'un ID unique à chaque entrée de l'index.
Conversion des types de données des colonnes.
Enregistrement du GeoDataFrame contenant l'index spatial dans un fichier shapefile.

5. Journalisation :

Enregistrement des informations d'exécution (date, heure, durée) dans un fichier journal.

Auteur : Mathias Chabal - INRS
"""

import warnings
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box
import laspy as lp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from neptune.tools import crs_river
from neptune.geometry import transform_coordinates
from neptune.logger import Logger
import os

if lp.__version__ <= '2.3.0':
    warnings.warn("La lecture des SRC risque d'être faux car, vôtre version de laspy est trop ancienne.")

warnings.filterwarnings('ignore')

### Paramètres à modifier ###
river_directory = Path(r"D:\CPD")  # Dossier de travail
lidar_directory = Path(r'D:\CPD\Lidar') # Dossier contenant les fichiers LIDAR
out_brut_index_file = Path(r'D:\CPD\Lidar\Index\Index_LB_CPD.shp') # Chemin de sortie pour l'index spatial

out_brut_index_file.parent.mkdir(parents=True, exist_ok=True)

def process_lidar_file(file_path: Path, target_crs: int):
    """
    Traite un fichier LAS/LAZ et renvoie un dict avec les infos pour l’index.
    """
    with lp.open(file_path) as lf:
        header = lf.header

    file_crs_obj = header.parse_crs()
    file_crs = file_crs_obj.to_epsg() if file_crs_obj is not None else None

    bbox_poly = box(
        header.x_min,
        header.y_min,
        header.x_max,
        header.y_max
    )

    if file_crs:
        if file_crs != target_crs:
            bbox_poly = transform_coordinates(bbox_poly, file_crs, target_crs)
    else:
        # si pas de CRS, on suppose qu’il est déjà dans le CRS cible
        file_crs = target_crs

    if not bbox_poly.is_valid:
        raise ValueError(f"L'emprise du fichier {file_path} n'est pas valide.")

    point_density = header.point_records_count / bbox_poly.area

    return {
        "NOM_FICHIER": file_path.name,
        "Chemin": str(file_path),
        "Densite": point_density,
        "ORI_CRS": file_crs,
        "geometry": bbox_poly,
    }


def main():

    river_code = river_directory.name
    log = Logger(river_code, river_directory)
    CRS = crs_river(river_code)
    lidar_folder = lidar_directory

    file_list = list(lidar_folder.glob('*.la*'))

    n_workers = os.cpu_count() -1
    results = []
    with ProcessPoolExecutor(n_workers) as executor:
        future_to_file = {
            executor.submit(process_lidar_file, file, CRS): file
            for file in file_list
        }

        for future in tqdm(as_completed(future_to_file),
                           total=len(future_to_file),
                           desc="Création de l'index spatial"):

            res = future.result()
            results.append(res)

    zone_index = gpd.GeoDataFrame(
        results,
        columns=['NOM_FICHIER', 'Chemin', 'Densite', 'ORI_CRS', 'geometry'],
        geometry='geometry',
        crs=CRS
    )

    zone_index = zone_index.reset_index(drop=True)
    zone_index['ID'] = zone_index.index

    print(f"Enregistrement de l'index spatial brut dans : {out_brut_index_file}")
    zone_index.to_file(out_brut_index_file, engine="pyogrio", use_arrow=True)

    log.stop()
    log.print_log()


if __name__ == "__main__":
    main()

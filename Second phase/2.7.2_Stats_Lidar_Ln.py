"""
SERIF-TOOLBOX - Étape 2
2.7.2 Statistique lidar brut tributaires

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

Code info : Ce code Python traite des données de transects polygonnaux et des données LIDAR pour calculer l'altitude
minimale (LB_MIN) de chaque transect et extraire des statistiques zonales à partir d'un raster de FACC (Flow
Accumulation).

1. Chargement et préparation des données :

Définition des chemins vers les données (transects, fichiers LIDAR, raster FACC) et du répertoire de travail.
Lecture des données de transects.
Reprojection des transects si nécessaire.
Création d'un dictionnaire pour stocker les erreurs.

2. Traitement par tributaire :

Boucle sur chaque tributaire unique.
Lecture des données LIDAR (fichiers .laz) avec laspy.

3. Traitement par transect :

Boucle sur chaque transect du tributaire courant. Vérification du type de géométrie du transect (doit être un
polygone valide) et de la méthode (doit être 'Linear'). Création d'un objet matplotlib.path.Path à partir des
coordonnées du polygone du transect pour identifier les points LIDAR à l'intérieur du polygone du transect.
Sélection des points LIDAR à l'intérieur du polygone.
Calcul de l'altitude minimale (LB_MIN) parmi les points LIDAR sélectionnés et stockage dans le GeoDataFrame
des transects.
Gestion des erreurs potentielles lors du traitement de chaque transect.

4. Calcul des statistiques zonales :

Utilisation de rasterstats.gen_zonal_stats pour calculer des statistiques (ici, le maximum) à partir du raster FACC
pour chaque polygone de transect. Création d'un nouveau GeoDataFrame à partir des résultats des statistiques zonales.

5. Nettoyage et enregistrement final :

Sélection des colonnes nécessaires et conversion des types de données.
Enregistrement du GeoDataFrame final (avec les statistiques zonales) dans le fichier shapefile de sortie.

7. Gestion des erreurs et journalisation :

Affichage des erreurs rencontrées pendant le traitement.
Enregistrement des informations d'exécution (date, heure, durée) dans un fichier journal.

Auteur : Mathias Chabal - INRS
"""

import geopandas as gpd
import laspy as lp
from tqdm import tqdm
import numpy as np
from pathlib import Path
import traceback
from neptune.tools import crs_river, get_columns_ln, get_col_dtype_ln
from neptune.logger import Logger
from shapely import contains_xy
from exactextract import exact_extract

### Paramètres à modifier ###
river_directory = Path(r'D:\CPD') # Dossier de travail
lidar_files_dir = Path(r'D:\CPD\Lidar\Merged') # Dossier contenant les fichiers LIDAR
facc_path = Path(r"D:\CPD\FACC_CPD_ProjectRaster.tif") # Raster d'accumulation des flux
transects_path = Path(r'D:\CPD\Lineaire\Transects_Ln_CPD.shp') # Transects linéaire
out_file = Path(r'D:\CPD\Lineaire\Transects_Ln_N1_CPD.shp') # Fichier de sortie

river_name = river_directory.name
log = Logger(river_name, river_directory)

CRS = crs_river(river_name)

if not facc_path.exists():
    raise FileNotFoundError(f"The file {facc_path} does not exist.")

transects = gpd.read_file(transects_path, engine="pyogrio", use_arrow=True)

if transects.crs.to_epsg() != CRS:
    transects.to_crs(CRS, inplace=True)

out_file.parent.mkdir(exist_ok=True)

print(f'Treating river : {river_name}')

n_unique_trib = len(transects['Nom'].unique())
# transects['LB_MIN'] = np.nan  # Initialisation de la colonne pour stocker les valeurs minimales d'altitude

error_dict = {}  # Dictionnaire pour stocker les erreurs rencontrées lors du traitement
trib_nb = 1

for trib_name, trib in transects.groupby("Nom"):

    print(f"\nTributary {trib_nb}/{n_unique_trib}")

    lidar_file = lidar_files_dir.joinpath(f'Merged_{trib_name}.laz')
    if not lidar_file.exists():
        error_dict[f"{trib_name}"] = f'The file {lidar_file} does not exist.'
        print(f'The file {lidar_file} does not exist.')
        continue  # Passe au tributaire suivante si le fichier lidar n'existe pas

    lidar_data = lp.read(lidar_file).xyz  # Lecture des données lidar (x, y, z)

    for i, poly in tqdm(trib["geometry"].items(), total=trib.shape[0], desc=f'Processing tributary {trib_name}'):

        try:
            # Vérification du type et de la validité du polygone, et de la méthode
            if not poly.is_valid:
                print(f'Invalid geometry for transect {i}')
                continue  # Passe au transect suivant si les conditions ne sont pas remplies

            minx, miny, maxx, maxy = poly.bounds
            bbox_mask = ((lidar_data[:, 0] >= minx) & (lidar_data[:, 0] <= maxx) &
                         (lidar_data[:, 1] >= miny) & (lidar_data[:, 1] <= maxy))
            clipped_lidar = lidar_data[bbox_mask]
            mask = contains_xy(poly, clipped_lidar[:, 0], clipped_lidar[:, 1])  # garde les dimensions x et y

            if np.sum(mask) == 0:
                continue  # Passe au transect suivant si aucun point lidar n'est trouvé à l'intérieur du polygone

            masked_lidar_data = clipped_lidar[mask]
            transects.at[i, 'LB_MIN'] = masked_lidar_data[:, 2].min()  # Enregistrement de l'altitude minimale des
            # points lidar à l'intérieur du polygone

        except Exception as error:
            error_message = traceback.format_exception_only(type(error), error)[-1].strip()
            print('error')
            error_dict[f"{trib_name}--{i}"] = error_message

    trib_nb += 1

transects.to_file(out_file, engine="pyogrio", use_arrow=True)

# Calcul des statistiques zonales avec rasterstats
transects["FACC_max"] = exact_extract(str(facc_path), transects, ["max"], output="pandas", progress=True)

columns = transects.columns
# Garde seulement les colonnes nécessaires et modifie le type si mauvais

transects = transects[get_columns_ln(columns)].astype(get_col_dtype_ln(columns), errors='ignore')

transects.to_file(out_file, engine="pyogrio", use_arrow=True)

if error_dict:
    print('\nFichier sauvegardé avec des erreurs :\n', error_dict)

log.stop()
log.print_log()

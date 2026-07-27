"""
SERIF-TOOLBOX - Étape 2
2.1 Transects niveau 1

Info de development :

Python version : 3.11.8
virtual environment : conda
numpy : 1.24.3
geopandas version : 1.0.1
pandas version : 2.0.3
arcpy version : 3.3
ArcGis Pro version : 3.3.0

Info code : Ce code Python traite des données de transects fluviaux pour calculer la largeur maximale, identifier le
chenal principal, corriger l'accumulation des flux, et identifier les chenaux secondaires.

1. Chargement et préparation des données :

Récupération des chemins vers les données (transects, transects d'eau, raster FACC, ligne médiane) et des paramètres
(valeur C pour Q2) depuis ArcGIS. Lecture des données (transects, transects d'eau, ligne centrale) avec geopandas.
Reprojection des données dans un SCR unique si nécessaire.

2. Calcul de la largeur maximale :

Définition d'une fonction calculate_max_width qui calcule la largeur maximale d'un polygone en utilisant son
rectangle englobant orienté. Application de cette fonction aux géométries des transects active et des transects d'eau
pour calculer les largeurs maximales (stockées dans les colonnes 'MAX_WIDTH' et 'WAT_WIDTH' respectivement).

3. Statistiques zonales pour FACC :

Calcul des statistiques zonales (maximum) du raster FACC pour chaque transect à l'aide de rasterstats.gen_zonal_stats.
Ajout de la valeur maximale de FACC à chaque transect.

4. Identification du chenal principal :

Identification des transects du chenal principal en utilisant l'intersection avec la ligne centrale.
Création de DataFrames distincts pour les transects du chenal principal et les autres.

5. Correction de l'accumulation des flux (FACC) :

Tri des transects du chenal principal par PK.
Suppression des valeurs aberrantes de FACC (inférieures à 1 000 000).
Correction et interpolation des valeurs de FACC.
Calcul du débit Q2 à partir de FACC corrigé et de la valeur C.

6. Identification des chenaux secondaires :

Identification des transects appartenant aux chenaux secondaires (ni backwater, ni lac).
Regroupement des transects connectés en utilisant la relation de "touches" entre les polygones.
Attribution d'un identifiant unique à chaque groupe de transects connectés (chenal secondaire).

7. Journalisation :

Enregistrement des informations d'exécution (date, heure, durée) dans un fichier journal.

Auteur : Mathias Chabal - INRS
"""


from pathlib import Path
import geopandas as gpd
import pandas as pd
from exactextract import exact_extract
from neptune import tools
from neptune.corrections import correct_facc
from neptune.logger import Logger
from neptune.network import HNetworkSurface
import numpy as np

transects = Path(r"D:\CPD\Transects_CPD.shp")  # Chemin vers les transects
transects_water = Path(r"D:\CPD\Transects_Eau_CPD.shp")  # Chemin vers les transects d'eau
facc = r"D:\CPD\FACC_CPD_ProjectRaster.tif" # Chemin vers le raster FACC
centerline_path = Path(r"D:\CPD\Ligne_CPD.shp")# Chemin vers la ligne centrale eau
C_VAL_Q2 = float(0.0000455899)  # Valeur C pour le calcul de Q2
out_level1_path = Path(r"D:\CPD\Transects_N1_CPD.shp")  # Chemin de sortie pour les transects d'eau


river_directory = transects.parent
river_code = transects.parent.name  # Nom de la rivière (déduit du chemin des transects)
log = Logger(river_code, river_directory)

CRS = tools.crs_river(river_code)  # Système de coordonnées de référence (SCR) pour la rivière
transects_gdf = gpd.read_file(transects, engine="pyogrio", use_arrow=True)  # Lecture des transects
transects_water_gdf = gpd.read_file(transects_water, engine="pyogrio", use_arrow=True)  # Lecture des transects d'eau
centerline = gpd.read_file(centerline_path, engine="pyogrio", use_arrow=True)  # Lecture de la ligne médiane

transects_gdf.sort_values(by="PK", ignore_index=True, inplace=True)
transects_water_gdf.sort_values(by="PK", ignore_index=True, inplace=True)

# Vérification et projection des SCR
if transects_gdf.crs.to_epsg() != CRS:
    transects_gdf.to_crs(CRS, inplace=True)

if transects_water_gdf.crs.to_epsg() != CRS:
    transects_water_gdf.to_crs(CRS, inplace=True)

if centerline.crs.to_epsg() != CRS:
    centerline.to_crs(CRS, inplace=True)

print('Statistiques zonales')
transects_gdf['FACC_M2'] = exact_extract(str(facc), transects_gdf, ["max"], output="pandas", progress=True)

print('Fin des statistiques zonales')

transects_gdf['MAX_WIDTH'] = transects_gdf.geometry.minimum_bounding_radius()*2
transects_water_gdf['WAT_WIDTH'] = transects_water_gdf.geometry.minimum_bounding_radius()*2

# Identification du chenal principal
intersect_result = transects_gdf.intersects(centerline.union_all())
transects_gdf['Principal'] = intersect_result  # Marque les transects du chenal principal

print("Corection de l'accumulation des flux")

transects_gdf.loc[transects_gdf['FACC_M2'] < 1_000_000, 'FACC_M2'] = np.nan
FACC2 = transects_gdf.loc[transects_gdf["Principal"], 'FACC_M2'].copy()
facc_index = FACC2.index
FACC2.iloc[-1] = FACC2.min()  # Initialisation de la première valeur de FACC
REF = FACC2.iloc[-1]  # Valeur de référence pour l'accumulation des flux
facc_flipped = np.flip(FACC2.values)

# Correction de l'accumulation des flux
facc_corrected = correct_facc(facc_flipped, REF)

# Interpole les valeurs manquantes de FACC
facc_interpolated = pd.Series(np.flip(facc_corrected), index=facc_index).interpolate().bfill().ffill()
transects_gdf.loc[transects_gdf["Principal"], 'FACC_COR'] = facc_interpolated
transects_gdf.loc[transects_gdf["Principal"], 'Q2'] = C_VAL_Q2 * facc_interpolated ** 0.75  # Calcul du débit Q2

print('Traitement des chenaux')
graph = HNetworkSurface(transects_gdf)
graph.classify_channels()
out_transect_gdf = graph.distribute_discharge("Q2", "Q2_spli")

graph_path = river_directory.joinpath(f'Graphe/graphe_Sf_active_{river_code}.pkl')
graph_path.parent.mkdir(exist_ok=True)
graph.save_graph(graph_path)

print('Jointure spatiale')
transects_water_gdf = tools.sjoin_largest_overlap(transects_water_gdf, out_transect_gdf[['FACC_M2', 'FACC_COR',
                                                                                         'Q2', 'Q2_spli',
                                                                                         'MAX_WIDTH', 'geometry']])
print('Fin de la jointure spatial')

columns = transects_water_gdf.columns
transects_water_gdf = transects_water_gdf[tools.get_columns_sf(columns)].astype(tools.get_col_dtype_sf(columns),
                                                                                errors='ignore')
columns = out_transect_gdf.columns
out_transect_gdf = out_transect_gdf[tools.get_columns_sf(columns)].astype(tools.get_col_dtype_sf(columns),
                                                                          errors='ignore')

print('Sauvgarde des fichiers')

out_transect_gdf.to_file(transects, index=False, engine="pyogrio", use_arrow=True)

transects_water_gdf.to_file(out_level1_path, index=False, engine="pyogrio", use_arrow=True)

print(f'Sauvgardé comme : {transects}')
print(f'Sauvgardé comme : {str(out_level1_path)}')

print(f"Supression des géometries en double")

log.stop()
log.print_log()

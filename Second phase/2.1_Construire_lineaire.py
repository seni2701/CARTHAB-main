"""
SERIF-TOOLBOX - Étape 2
2.1 Regrouper les tributaires

Info de development :

Python version : 3.11.8
virtual environment : conda
numpy : 1.24.3
geopandas version : 1.0.1
pandas version : 2.0.3
arcpy version : 3.3
ArcGis Pro version : 3.3.0

Info code :

Ce code Python traite des données hydrographiques (GRHQ) pour préparer les lignes de tributaires pour des analyses
ultérieures.

1. Chargement et préparation des données :

Récupération des chemins vers le fichier GRHQ en entrée et le répertoire de travail depuis les paramètres ArcGIS.
Définition du nom de la rivière.
Suppression des fichiers intermédiaires potentiels.

2. Traitement des lignes :

Utilisation des outils ArcGIS pour :
Fusionner les lignes GRHQ en fonction du champ RTE_HORTON (UnsplitLine).
Lisser les lignes fusionnées (SmoothLine).
Inverser la direction des lignes (FlipLine).
Suppression des fichiers intermédiaires.

3. Traitement avec GeoPandas :

Lecture des lignes traitées avec geopandas.
Renommage de la colonne MAX_TOPONY en Toponyme.
Raccrochage des lignes de tributaires à la ligne principale à l'aide de la fonction snap_lines.
Attribution de noms uniques aléatoires à chaque tributaire.
Calcul de la longueur de chaque tributaire.
Filtrage des tributaires pour ne conserver que ceux dont la longueur est supérieure ou égale à 100 mètres.

4. Enregistrement des résultats :

Sélection des colonnes nécessaires et conversion des types de données.
Enregistrement des lignes de tributaires traitées dans un fichier shapefile.

5. Journalisation :

Enregistrement des informations d'exécution (date, heure, durée) dans un fichier journal.

Auteur : Mathias Chabal - INRS
"""

from pathlib import Path
import geopandas as gpd
from neptune.geometry import drop_z
from neptune.tools import random_alphanum, crs_river, get_columns_ln, get_col_dtype_ln
from neptune.network import *
from neptune.smoothing import smoothify
from neptune.logger import Logger


### Start of the code ###
grhq_path = Path(r"D:\CPD\temp_rhl.shp")  # GRHQ file path
river_directory = Path(r"D:\CPD")  # River directory path
out_file_name = Path(r"D:\CPD\Ligne_Pr_Tr_CPD.shp")

river_code = river_directory.name
log = Logger(river_code, river_directory)
CRS = crs_river(river_code)

print('Fusion des lignes')
gdf = gpd.read_file(grhq_path, engine="pyogrio", use_arrow=True)

if gdf.crs.to_epsg() != CRS:
    gdf.to_crs(CRS, inplace=True)

if gdf.has_z.any():
    gdf["geometry"] = gdf.geometry.apply(drop_z)

unsplited_lines = unsplit_lines(gdf, 'RTE_HORTON', {"TOPONYME": "mode"})

print('Lissage des lignes')

smoothed_lines = smoothify(unsplited_lines, smooth_iterations=5, num_cores=1)

print('Inverssement des lignes')
smoothed_lines["geometry"] = smoothed_lines["geometry"].reverse()
smoothed_lines: gpd.GeoDataFrame = smoothed_lines.rename(columns={'TOPONYME': 'Toponyme'})

# Calcule la longueur des entités
smoothed_lines['Longueur'] = smoothed_lines.length
smoothed_lines = smoothed_lines[smoothed_lines['Longueur'] >= 100]
smoothed_lines.sort_values(by="Longueur", ascending=False, inplace=True, ignore_index=True)

print('Fusion des lignes disconnectées')
merged_lines = merge_lines(smoothed_lines.explode(ignore_index=True))
merged_lines.reset_index(drop=True, inplace=True)

print('Correction des lignes qui se croisent')
corrected_lines = correct_crossing_lines(merged_lines.geometry)

print('Raccrochage des lignes')
snapped_lines = snap_lines(corrected_lines)

merged_lines['Longueur'] = snapped_lines.length

print("Assignation de la hierarchie aux lignes")
hierarchy = set_hierarchy(snapped_lines)

merged_lines["geometry"] = snapped_lines
merged_lines["Hierarchie"] = hierarchy
merged_lines['ID'] = merged_lines.index

print('Assignation des identifiants uniques aléatoires')
# Assigner un nom unique aléatoire au tributaire
length_df = len(merged_lines)
random_ids = random_alphanum(n=length_df, k_str=3, k_num=2, prefix=river_code)

merged_lines['Nom'] = random_ids

# Garde seulement les colonnes neccessaires et modifit le type de données
columns = merged_lines.columns
final_lines = merged_lines[get_columns_ln(columns)].astype(get_col_dtype_ln(columns), errors='ignore')

final_lines.to_file(out_file_name, engine="pyogrio", use_arrow=True)
print(f'Sauvegardé sous : {out_file_name}')

print('Division des lignes')
splited_lines = split_hydro_network(final_lines)

print('Création du graphe')
river_graph = HNetworkLinear(splited_lines)

if not river_graph.is_fully_connected:
    non_connected = river_graph.find_disconnected_components()
    print(f"Le graphe n'est pas entièrement connecté, voici les sections non connectées : {non_connected}")

graph_path = river_directory / f'Graphe/graphe_Ln_{river_code}.pkl'
graph_path.parent.mkdir(exist_ok=True)
river_graph.save_graph(graph_path)

print(f"Graphe sauvegardé sous : {graph_path}")

log.stop()
log = Logger(river_code, river_directory.parent)

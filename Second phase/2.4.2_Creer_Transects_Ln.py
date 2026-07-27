"""
SERIF-TOOLBOX - Étape 2
2.5.2 Création des transects linéaires

Development info :

Python version : 3.11.5
virtual environment : conda
Python IDE : PyCharm
geopandas version : 0.9.0
shapely version : 2.0.2
tqdm version : 4.66.1

Code info :

Ce code Python crée des transects perpendiculaires à une ligne centrale de rivière, en utilisant des diagrammes
de Voronoi pour la méthodologie Linéaire.

1. Chargement et préparation des données :

Définition des chemins vers les données (ligne centrale) et du répertoire de travail.
Détermination du système de coordonnées de référence (CRS) pour la rivière.
Reprojection de la ligne centrale si nécessaire.

2. Génération de points le long de la ligne centrale :

Utilisation de la fonction points_along_line pour créer des points équidistants (tous les 5 mètres) le long de la
ligne centrale. Les points sont regroupés par séquence (reach_ID) et chaque séquence correspond à un tributaire de la
rivière.

3. Création des polygones de Voronoi :

Pour chaque séquence unique de points :
Création d'un tampon autour de la sous-ligne centrale correspondante.
Utilisation de la fonction voronoi_diagrams pour créer des polygones de Voronoi à partir des points de la séquence, en
étendant les polygones jusqu'au tampon.
Intersection des polygones de Voronoi avec le tampon pour ne conserver que les parties à l'intérieur du tampon.
Stockage des polygones de Voronoi intersectés dans un GeoDataFrame.
Gestion des erreurs potentielles lors de la création des polygones de Voronoi pour chaque branche.

4. Nettoyage et enregistrement des données :

Attribution d'un identifiant unique à chaque polygone de Voronoi.
Filtrage des polygones de Voronoi pour supprimer ceux qui ont causé des erreurs lors de leur création.
Filtrage des polygones de Voronoi pour ne conserver que ceux qui sont des polygones simples et non vides.
Sélection des colonnes nécessaires et conversion des types de données.
Enregistrement des polygones de Voronoi (transects) dans un fichier shapefile.

5. Journalisation :

Enregistrement des informations d'exécution (date, heure, durée) dans un fichier journal.

Auteur : Mathias Chabal - INRS
"""

import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
import numpy as np
from neptune.tools import crs_river, get_columns_ln, get_col_dtype_ln
from neptune.geometry import points_along_line, voronoi_diagrams
from neptune.logger import Logger
from neptune.network import HNetworkLinear

### Paramètres à modifier ###
river_directory = Path(r'D:\CPD')  # Dossier de travail
centerline_path = Path(r'D:\CPD\Ligne_Pr_Tr_CPD.shp') # Ligne centrale des tributaires et principale
water_file = Path(r'D:\CPD\Eau_ALL_CPD.shp') # Masque eau
rhs_path = Path(r'D:\CPD\rhs_CPD.shp') # Couche RHS (lac)
out_file = Path(r'D:\CPD\Lineaire_all\Transects_Ln_CPD.shp') # Fichier de sortie

### Début du code ###
if not rhs_path.exists():
    raise FileNotFoundError(f"The file {rhs_path} does not exist.")

river_code = river_directory.name
log = Logger(river_code, river_directory)

# Obtenez le système de coordonnées de référence (CRS) pour la rivière à partir de son code
CRS = crs_river(river_code)

centerline = gpd.read_file(centerline_path, engine="pyogrio", use_arrow=True)
water = gpd.read_file(water_file, engine="pyogrio", use_arrow=True)

# Verifiez si le CRS du fichier est le même que CRS de la rivière. Si ce n'est pas le cas, mettez-le à jour.
if centerline.crs.to_epsg() != CRS:
    centerline.to_crs(CRS, inplace=True)

if water.crs.to_epsg() != CRS:
    water.to_crs(CRS, inplace=True)

centerline = centerline[centerline['Methode'] != 'Surfacique']

lines = centerline.geometry.difference(water.union_all())
for i, line in zip(lines.index, lines):
    if line.geom_type == 'MultiLineString':
        lengths = []
        for geom in line.geoms:
            lengths.append(geom.length)

        idx = np.argmax(lengths)
        lines.loc[i] = line.geoms[idx]

centerline["geometry"] = lines

graph_path = Path(r"D:\CPD\Graphe\graphe_Ln_CPD.pkl")
graph = HNetworkLinear.from_file(graph_path)
graph_gdf = graph.gdf_input

sec_pos_mask =  graph_gdf.reach_ID.apply(lambda x: x.split("_")[-1]).astype(int) == 0
hierarchy_mask = graph_gdf.Hierarchie == 1
first_trib_sec = graph_gdf[sec_pos_mask & hierarchy_mask][["reach_ID", "Nom"]]

for i, r in first_trib_sec.iterrows():
    tribs = graph.get_tributaries(r.reach_ID)
    if tribs:
        centerline.loc[centerline["Nom"].isin(tribs), "SUB_BV_ID"] = r.Nom
    else:
        centerline.loc[centerline["Nom"] == r.Nom, "SUB_BV_ID"] = r.Nom

name = centerline.loc[centerline["Hierarchie"] == 0, "Nom"].iloc[0]
centerline.loc[centerline["Hierarchie"] == 0, "SUB_BV_ID"] = "0" + name

# Générez des points le long de la ligne centrale à chaque 5 mètres
points_transects = points_along_line(centerline, 5, centerline.crs, keep_cols=True)

# Dupliquez le géodataframe de points et remplacez la colonne de géométrie par None
voronoi_polys = points_transects.copy()
voronoi_polys['geometry'] = None

# Obtenez une liste de séquences uniques de création des transects par tributaire
unique_seq = points_transects['SEQ_ID'].unique()
error_list = []

# Boucle sur chaque séquence unique pour créer des polygones de Voronoi
for u in tqdm(unique_seq, total=len(unique_seq), desc='Création des polygones de Voronoi'):

    # Obtenez tous les points appartenant à la même séquence
    sub_points = points_transects[points_transects['SEQ_ID'] == u]

    # Obtenez l'index min et max pour la sous-séquence de points
    min_index = sub_points.index.min()
    max_index = sub_points.index.max()

    # Obtenez la sous-ligne centrale qui correspond au nom de la sous-séquence
    sub_centerline = centerline[centerline['Nom'] == sub_points.at[min_index, 'Nom']]

    # Obtenez le nom de la branche de la sous-séquence
    name_trib = sub_points.at[min_index, 'Nom']

    try:
        # Créez un tampon autour de la sous-ligne centrale et créez des polygones de Voronoi à l'intérieur du tampon
        line_buffer = sub_centerline.union_all().buffer(15)

        # Créez des polygones de Voronoi à l'aide des points et étendez les polygones pour couvrir le tampon
        vor_polys = voronoi_diagrams(sub_points, extend_to=line_buffer, return_list=False)

        # Intersectez les polygones de Voronoi et le tampon pour ne garder que les parties se trouvant à l'intérieur
        # du tampon
        vor_polys['geometry'] = vor_polys.intersection(line_buffer)

        # Enregistrez les polygones de Voronoi intersectés dans le dataframe voronoi_polys
        voronoi_polys.loc[min_index:max_index, 'geometry'] = vor_polys['geometry'].to_list()

    except:
        # Si une erreur se produit pendant le processus, ajoutez le nom de la branche à la liste des erreurs
        error_list.append(name_trib)

        print(f'La branche {name_trib} n’a pas fonctionné, on passe à la suivante.')
        continue

# Attribuez un ID unique à chaque polygone de Voronoi
voronoi_polys['ID'] = list(range(len(voronoi_polys)))

print("Nettoyage et sauvegarde des données")

# Filtrer les polygones de Voronoi pour ne garder que ceux dont le nom n'est pas dans la liste des erreurs
voronoi_polys = voronoi_polys[~voronoi_polys['Nom'].isin(error_list)]

# Filtrer les polygones de Voronoi pour ne garder que ceux qui ne sont pas des GeometryCollections et qui ne sont pas
# vides
voronoi_polys = voronoi_polys[(voronoi_polys['geometry'].geom_type != 'GeometryCollection') & ~voronoi_polys['geometry'].is_empty]

voronoi_polys = voronoi_polys[~((voronoi_polys['PK'] == 0) & (voronoi_polys['Hierarchie'] < 2))]
voronoi_polys.loc[voronoi_polys['Hierarchie'] < 2, 'PK'] = voronoi_polys.loc[voronoi_polys['Hierarchie'] < 2, 'PK'] - 5

voronoi_polys = voronoi_polys[~((voronoi_polys['PK'] <= 15) & (voronoi_polys['Hierarchie'] > 1))]
voronoi_polys.loc[voronoi_polys['Hierarchie'] > 1, 'PK'] = voronoi_polys.loc[voronoi_polys['Hierarchie'] > 1, 'PK'] - 20

print("Assignattion de l'atribut reach_ID")
# Assigner l'atribut reach_ID
for reach, df in graph_gdf.groupby("reach_ID")[["geometry", "Nom"]]:

    sub_tr = voronoi_polys[voronoi_polys["Nom"] == df["Nom"].iloc[0]]
    mask = sub_tr.intersects(df["geometry"].iloc[0])

    if mask.any():
        voronoi_polys.loc[sub_tr[mask].index, "reach_ID"] = reach

if voronoi_polys.reach_ID.isna().any():
    print("Certain transects n'ont pas de reach_ID")

print("Assignation de l'atribute Lac")
rhs = gpd.read_file(rhs_path, engine="pyogrio", use_arrow=True).to_crs(CRS)
rhs_poly = rhs[rhs["TYPECE"] == 21].union_all()

voronoi_polys.reset_index(inplace=True, drop=True)
water_lake_mask = voronoi_polys.sindex.query(rhs_poly, predicate="intersects", output_format="dense")
voronoi_polys["Lac"] = water_lake_mask

columns = voronoi_polys.columns
voronoi_polys = voronoi_polys[get_columns_ln(columns)].astype(get_col_dtype_ln(columns), errors='ignore')

if not out_file.parent.exists():
    out_file.parent.mkdir()

voronoi_polys.to_file(out_file, engine="pyogrio", use_arrow=True)
print(f'Saved as : {out_file}')

if not len(error_list) == 0:
    print(error_list)

log.stop()
log.print_log()

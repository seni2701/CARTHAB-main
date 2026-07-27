"""
SERIF-TOOLBOX - Étape 2
2.8.2 Corrections et ajouts des débits, pentes

Info de development :

Python version : 3.9.18
virtual environment : conda
numpy : 1.20.1
geopandas version : 0.14.0
shapely version : 2.0.1
scipy version : 1.6.2
pandas version : 1.4.4
matplotlib version : 3.6.0
ap version : 3.1
ArcGis Pro version : 3.2.2

Info code : Ce code Python effectue une analyse hydrologique et sédimentaire d'une rivière à partir de données de
transects, de données LIDAR brutes d'élévation, de la ligne médiane de la rivière, d'un index d'ortho-images et de
valeurs de 'c' pour le calcul du débit. Voici un résumé méthodologique des principales étapes :

1. Chargement et préparation des données :

Lecture des données depuis des fichiers shapefile (transects, données LIDAR brutes, ligne médiane) et un fichier
Excel (valeurs de 'c').

Reprojection des données dans un système de coordonnées de référence (SCR) unique.
Calcul de la surface des transects et des données LIDAR.
Fusion des données de transects et des données LIDAR.
Correction des valeurs aberrantes et interpolation des valeurs manquantes pour l'accumulation des flux (FACC)
et l'élévation.

2. Calcul des paramètres hydrauliques :

Calcul de la pente locale et de la largeur médiane des transects.
Calcul de la pente sur 10 fois la largeur et de la largeur médiane sur 10 fois la largeur.

3. Estimation des débits :

Jointure spatiale des transects avec l'index des ortho-images pour associer les dates des images aux transects.
Calcul du débit image (Q_IMG) à partir des valeurs de 'c', de l'accumulation des flux corrigée (FACC_COR) et d'une
relation empirique.

4. Traitement des chenaux :

Identification et numérotation des chenaux principaux et secondaires à l'aide de la ligne médiane et de critères
géométriques (transects qui se touchent).
Correction des valeurs d'élévation pour chaque chenal.
Calcul de la pente pour chaque chenal (pente globale si le chenal a moins de 14 transects, pente locale sinon).

5. Répartition du débit :

Calcul des proportions de débit pour chaque transect en fonction des ortho-images.
Lissage des proportions de débit.
Calcul du débit réparti (Q_IMG_spli) en multipliant le débit image par les proportions lissées.

6. Calcul de la puissance spécifique et prédiction du D50 et D84 :

Calcul de la puissance spécifique (PS) à partir du débit, de la pente et de la largeur des transects. Prédiction du
D50 (diamètre médian des sédiments) à l'aide d'un modèle de régression linéaire.
Calcul du D84 à partir de la puissance spécifique et d'une relation empirique.

7. Détermination de la limite de montaison :

Identification des tronçons de rivière inaccessibles en fonction de la différence d'altitude entre les transects.Le
gain en élévation sur 100m ne doit pas être supérieure à 12m.

8. Nettoyage et sauvegarde des données :

Sélection des colonnes pertinentes et conversion des types de données.
Suppression des géométries dupliquées.
Sauvegarde des résultats dans un fichier shapefile.

9. Création de graphiques de validation (optionnel) :

Création de graphiques interactifs (HTML) pour visualiser l'accumulation des flux et le profil en long de la rivière.
Gestion des anciens graphiques (déplacement et renommage).

10. Enregistrement des informations d'exécution :

Enregistrement de la date, de l'heure et de la durée d'exécution du script dans un fichier journal.

Auteur : Mathias Chabal - INRS
"""

from pathlib import Path
import warnings
from shutil import move

from scipy.stats import linregress
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

from neptune.network import HNetworkSurface
from neptune.logger import Logger
from neptune.tools import crs_river, get_columns_sf, get_col_dtype_sf, sjoin_largest_overlap
from neptune.corrections import correct_z
from neptune.slope import calculate_local_slope, calculate_adaptive_slope_width


tr_n1_path = Path(r"D:\CPD\Transects_N1_CPD.shp")
elev_brut_path = Path(r"D:\CPD\Transects_LB_CPD.shp")
centerline_path = Path(r"D:\CPD\Ligne_CPD.shp")
index_img_path = Path(r"D:\CPD\PHOTOS\Index\index_ortho_CPD_reel.shp")
c_img_file = Path(r"D:\CPD\Debit\Donnees_debits_CPD.xlsx")
outfile_name = Path(r"D:\CPD\Transect_N2_CPD.shp")
make_validation_plots = True

# Chargement et préparation des données
river_directory = tr_n1_path.parent
river_code = river_directory.name
log = Logger(river_code, river_directory, graphique_validation=make_validation_plots)

CRS = crs_river(river_code)  # Système de coordonnées de référence (SCR) de la rivière

gdf = gpd.read_file(tr_n1_path, engine="pyogrio", use_arrow=True)
gdf.sort_values(by=['PK'], inplace=True, ignore_index=True)
gdf['Surface'] = gdf.area  # Calcul de la surface des transects

print(r"Préparation des données.")

if gdf.crs.to_epsg() != CRS:
    gdf.to_crs(CRS, inplace=True)  # Reprojection des transects si nécessaire

centerline = gpd.read_file(centerline_path, engine="pyogrio", use_arrow=True)  # Lecture de la ligne médiane

if centerline.crs.to_epsg() != CRS:
    centerline.to_crs(CRS, inplace=True)  # Reprojection de la ligne médiane si nécessaire

gdf['PK'] = gdf['PK'].astype(int, errors='ignore')  # Conversion des PK en entiers

df_brut = gpd.read_file(elev_brut_path, engine="pyogrio", use_arrow=True)  # Lecture des données lidar brutes d'élévation

if df_brut.crs.to_epsg() != CRS:
    df_brut.to_crs(CRS, inplace=True)  # Reprojection des données lidar si nécessaire

df_brut = df_brut[['ID', 'LB_Q25']]  # Sélection des colonnes ID et LB_Q25 (altitude)
df_brut['ID'] = df_brut['ID'].astype(int, errors='ignore')

gdf = gdf.merge(df_brut, on='ID')  # Fusion des DataFrames df_ori et df_brut sur la colonne 'ID'

gdf['Principal'] = gdf.intersects(centerline.union_all())

print(r"Correction des valeurs d'élévation.")

z = gdf.loc[gdf["Principal"], "LB_Q25"]  # Copie des valeurs d'élévation (LB_Q25)
z_index = z.index
z = np.flip(z.replace(0, np.nan).values)
REF = np.nanmax(z[:6])

z_corrected = correct_z(z, REF)

z_interpolated = pd.Series(np.flip(z_corrected), index=z_index).interpolate().bfill().ffill()
gdf.loc[gdf["Principal"], "LB_Q25_COR"] = z_interpolated

print(r'Calcul des valeurs locales de pente.')

pk = gdf.loc[gdf["Principal"], "PK"]
slope = calculate_local_slope(pk, z_interpolated)
gdf.loc[gdf["Principal"], "Pente"] = pd.Series(slope, index=z_index).interpolate().bfill().ffill()

# Calcul de la largeur médiane sur 7 transects (35m)
gdf.loc[gdf["Principal"], 'W_med'] = gdf.loc[gdf["Principal"], 'MAX_WIDTH'].rolling(7, center=True).median().bfill().ffill()

max_width = gdf.loc[gdf["Principal"], "MAX_WIDTH"]
pk = gdf.loc[gdf["Principal"], "PK"]
elev = gdf.loc[gdf["Principal"], "LB_Q25_COR"]
w_med = gdf.loc[gdf["Principal"], "W_med"]
ref_index = w_med.index
slope_10, m_width_10 = calculate_adaptive_slope_width(pk, elev, w_med, max_width)

gdf.loc[gdf["Principal"], "Pente_10"] = pd.Series(slope_10, index=ref_index).interpolate().bfill().ffill()
gdf.loc[gdf["Principal"], "M_WIDTH_10"] = pd.Series(m_width_10, index=ref_index).interpolate().bfill().ffill()

print(r'Estimation des débits Q2, Q Lidar et Q Ortho')

ortho_index = gpd.read_file(index_img_path, engine="pyogrio", use_arrow=True)
ortho_index.to_crs(CRS, inplace=True)

gdf_with_dates = sjoin_largest_overlap(gdf, ortho_index)

c_img_df = pd.read_excel(c_img_file, sheet_name='valeurs_c')
c_img_df['date'] = pd.to_datetime(c_img_df['date']).dt.date

# Harmoniser les différents noms de colonnes présents potentiellement dans les index ortho
gdf_with_dates = gdf_with_dates.rename(columns={'DATE_SOURC': 'DATE_IMG', 'DATE_PHOTO': 'DATE_IMG',
                                                '_DATE_IMG': 'DATE_IMG'})

gdf_with_dates['DATE_IMG'] = gdf_with_dates['DATE_IMG'].str.replace('/', '-')  # Uniformisation du format des dates

gdf_with_dates['DATE_IMG'] = pd.to_datetime(gdf_with_dates['DATE_IMG']).dt.date

for index, row in c_img_df.iterrows():
    date_result = gdf_with_dates['DATE_IMG'] == row.date

    if date_result.any():
        gdf_with_dates.loc[date_result, 'c_val_img'] = row.valeur_c
    else:  # Si aucune correspondance n'est trouvée
        raise ValueError(f"Erreur date : Il n'y a pas de correspondance entre les dates en entrées dans le " 
                        f"fichier et les dates de l'index ortho, ou les formats ne sont pas indentiques (le format "
                        f"doit-être : "
                        f"yyyy-mm-dd).\nDate index : {gdf_with_dates['DATE_IMG'].unique()}\nDate : {str(row.date)}")


gdf_with_dates.loc[gdf_with_dates["Principal"], 'Q_IMG'] = (gdf_with_dates.loc[gdf_with_dates["Principal"], 'c_val_img'] *
                                                 gdf_with_dates.loc[gdf_with_dates["Principal"], 'FACC_COR'] ** 0.75)

print('Traitement des chenaux')
graph = HNetworkSurface(gdf_with_dates)
graph.classify_channels()

gdf_with_q = graph.distribute_discharge("Q_IMG", "Q_IMG_spli")

graph_path = river_directory.joinpath(f'Graphe/graphe_Sf_eau_{river_code}.pkl')
graph_path.parent.mkdir(exist_ok=True)
graph.save_graph(graph_path)

for u, sub_channel in gdf_with_q.groupby("CHL_ID"):  # Boucle sur chaque identifiant de chenal unique.
    if u == 'P' or u == 'n':  # Si l'identifiant est "n" (non attribué), on passe à l'identifiant suivant.
        continue

    z = sub_channel["LB_Q25"]  # Copie des valeurs d'élévation (LB_Q25)
    z_index = z.index
    z = np.flip(z.replace(0, np.nan).values)

    # Exclue les chenaux entièrement nuls et qui ont moins de 4 valeurs valides s'ils ont au moins 4 transects.
    if np.all(np.isnan(z)) or (np.sum(~np.isnan(z)) <= 3 < len(z)):
        warnings.warn(f"Le chenal '{u}' a moins de 4 ou aucune valeurs valides dans la colonne LB_Q25.")
        continue

    REF = np.nanmax(z[:6])

    if np.isnan(REF):
        REF = np.nanmax(z)

    z_coorected = correct_z(z, REF)

    z_interpolated = pd.Series(np.flip(z_coorected), index=z_index).interpolate().bfill().ffill()
    gdf_with_q.loc[sub_channel.index, 'LB_Q25_COR'] = z_interpolated


    PK = sub_channel['PK'].copy()
    if sub_channel.shape[0] < 14:  # Si le chenal a moins de 14 transects, on calcule la pente globale du chenal.
        slope, intercept, r_value, p_value, std_err = linregress(PK, z_interpolated)
        gdf_with_q.loc[sub_channel.index, 'Pente'] = slope

    else:  # Si le chenal a 14 transects ou plus, on calcule la pente locale en chaque point.

        slope = calculate_local_slope(PK, z_interpolated)
        gdf_with_q.loc[sub_channel.index, "Pente"] = pd.Series(slope, index=z_index).interpolate().bfill().ffill()


# df, channels et b_l_transects et les trie par PK.  `ignore_index=True` réinitialise les index.
gdf_with_q['Duplique'] = gdf_with_q['PK'].duplicated(keep=False)  # Crée une colonne 'Duplicated' qui indique si un PK
# est dupliqué (True si oui, False sinon).

print(r'Calcule de la puissance spécifique et prédiction du D50 et D84')

# Remplace les valeurs NaN dans Slope_10 par les valeurs correspondantes dans Slope.
gdf_with_q.loc[gdf_with_q['Pente_10'].isna(), 'Pente_10'] = gdf_with_q.loc[gdf_with_q['Pente_10'].isna(), 'Pente']
# Remplace les valeurs NaN dans M_WIDTH_10 par les valeurs correspondantes dans MAX_WIDTH.
gdf_with_q.loc[gdf_with_q['M_WIDTH_10'].isna(), 'M_WIDTH_10'] = gdf_with_q.loc[gdf_with_q['M_WIDTH_10'].isna(),
                                                                                                'MAX_WIDTH']

gdf_with_q.loc[gdf_with_q['Pente_10'] == 0, 'Pente_10'] = 0.000001  # Remplace les valeurs nulles dans Slope_10 par une
# valeur très petite pour éviter les divisions par zéro.
gdf_with_q.loc[gdf_with_q['Pente'] == 0, 'Pente'] = 0.000001

# Calcul de la puissance spécifique (PS) :
gdf_with_q['PS'] = 999.98 * 9.8 * gdf_with_q['Q2_spli'] * gdf_with_q['Pente'] / gdf_with_q['MAX_WIDTH']
gdf_with_q['PS_10'] = 999.98 * 9.8 * gdf_with_q['Q2_spli'] * gdf_with_q['Pente_10'] / gdf_with_q['M_WIDTH_10']

gdf_with_q['D50'] = 4.87 * (gdf_with_q['PS_10'] ** 0.63)
gdf_with_q['D84'] = 7.516024639618281 * (gdf_with_q['PS_10'] ** 0.6787700491190716)  # Calcul de D84 à partir
# de PS_10 (relation empirique).

print(r'Netoyage des données')
df_col_names = gdf_with_q.columns  # Récupère les noms de colonnes.
# Sélectionne les colonnes spécifiées par get_columns et convertit les types de données.
final_gdf = gdf_with_q[get_columns_sf(df_col_names)].astype(get_col_dtype_sf(df_col_names), errors='ignore')

print(f"Sauvegarde des données dans l'environnement de travail sous le nom : {outfile_name}")
final_gdf.to_file(outfile_name, engine="pyogrio", use_arrow=True)

log.stop()

if make_validation_plots:  # Vérifie si la création de graphiques de validation est activée.
    print(r"Création des graphiques de validations")
    validation_plots_dir = outfile_name.parent.joinpath('Validation Plots')
    validation_plots_dir.mkdir(exist_ok=True)  # Crée le répertoire s'il n'existe pas.

    plot_df = final_gdf[final_gdf['Principal']]
    pk = plot_df['PK']

    # Création du graphique d'accumulation des flux :
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=pk, y=plot_df['FACC_COR'], mode='lines', name='FACC corrigé'))
    fig1.add_trace(go.Scatter(x=pk, y=plot_df['FACC_M2'], mode='lines', name='FACC brut'))
    fig1.update_layout(title='Accumulation des flux', xaxis_title='PK', yaxis_title='Accumulation des flux (m2)')

    # Création du graphique du profil en long (élévation) :
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pk, y=plot_df['LB_Q25_COR'], mode='lines', name='Elévation corrigée'))
    fig2.add_trace(go.Scatter(x=pk, y=plot_df['LB_Q25'], mode='lines', name='Elévation brute'))
    fig2.update_layout(title='Profil en long', xaxis_title='PK', yaxis_title='Élévation (m)')

    out_plot_FACC_path = validation_plots_dir.joinpath('flow_accumulation.html')
    out_plot_elev_path = validation_plots_dir.joinpath('elevation.html')

    # Gestion des anciens graphiques :
    if out_plot_FACC_path.exists() or out_plot_elev_path.exists():  # Vérifie si des graphiques existent déjà.
        new_plot_dir = f'{validation_plots_dir}/previous_plots'  # Crée un répertoire pour stocker les anciens graphiques.
        Path(new_plot_dir).mkdir(exist_ok=True)  # Crée le répertoire s'il n'existe pas.

        for file in Path(new_plot_dir).iterdir():  # Supprime les anciens graphiques dans le répertoire "previous_plots".
            if file.is_file():
                file.unlink()

        # Déplace et renomme les anciens graphiques.
        move(out_plot_FACC_path, new_plot_dir)
        Path(f'{new_plot_dir}/flow_accumulation.html').rename(f'{new_plot_dir}/flow_accumulation_old.html')

        move(out_plot_elev_path, new_plot_dir)
        Path(f'{new_plot_dir}/elevation.html').rename(f'{new_plot_dir}/elevation_old.html')

    # Enregistrement des graphiques :
    fig1.write_html(out_plot_FACC_path)
    fig2.write_html(out_plot_elev_path)
    print(f"Graphiques sauvegards dans le dossier : {validation_plots_dir}")

log.print_log()

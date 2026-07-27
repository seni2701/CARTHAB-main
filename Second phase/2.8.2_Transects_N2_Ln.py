"""
SERIF-TOOLBOX - Étape 2
2.8.3 Ajout et correction : élévation, pente, débit, surface drainée et accessibilité

Info de development :

Python version : 3.9.18
virtual environment : conda
numpy : 1.20.1
geopandas version : 0.14.0
shapely version : 2.0.1
scipy version : 1.6.2
pandas version : 1.4.4

Info code : Ce script Python vise à corriger les données altimétriques et d'accumulation des flux, calculer la pente du
lit mineur, et prédire la granulométrie du lit.

La méthodologie peut être résumée comme suit :

Préparation des données :

    Lecture des données : Le script lit les fichiers shapefile contenant les lignes centrales du cours d'eau (principal
    et tributaire), les polygones d'eau, et les transects (principaux et tributaires).

    Reprojection : Les données sont reprojetées dans un système de coordonnées de référence (CRS) unique pour la rivière.

    Création de la hiérarchie des cours d'eau : Une hiérarchie est établie pour distinguer le cours d'eau principal et
    ses tributaires, en utilisant des buffers et des intersections. Un identifiant unique de sous-bassin versant
    (SUB_BV_ID) est attribué à chaque segment.

    Nettoyage des géométries : Les géométries invalides dans les transects des
    tributaires sont corrigées ou supprimées.

Correction des données et calculs :

    Attribution de la hiérarchie et de l'ID de sous-bassin : Les informations de hiérarchie et d'ID de sous-bassin sont
    transférées des lignes centrales vers les transects.
    Correction de l'accumulation des flux (FACC_COR) : Les valeurs d'accumulation des flux (FACC_max) sont corrigées en
    utilisant un seuil basé sur le 90ème percentile et une interpolation. Les valeurs aberrantes sont supprimées.

    Correction des élévations (LB_MIN_COR) : Les élévations du lit mineur (LB_MIN) sont corrigées et interpolées. Une
    attention particulière est portée aux valeurs en début et fin de transect.

    Calcul de la pente (Slope) : La pente est calculée par régression linéaire sur une fenêtre de 7 points
    (3 de chaque côté du point central) des élévations corrigées.

    Vérification de l'accessibilité :
    L'accessibilité des transects est évaluée en comparant les élévations et en vérifiant les intersections avec les
    transects surfaciques. Le gain en élévation sur 100m ne doit pas être supérieure à 12m.

    Calcul des débits (Q2, Q_IMG) et de la largeur (WIDTH) : Les débits Q2 et Q_IMG sont calculés à
    partir de l'accumulation des flux corrigée, en utilisant des constantes spécifiques. La largeur est également
    calculée à partir de l'accumulation des flux.

    Prédiction de la largeur maximale (MAX_WIDTH) : Un modèle de régression chargé depuis un fichier pickle est
    utilisé pour prédire la largeur maximale à partir du débit Q2.

    Calcul de la puissance spécifique (PS) : La puissance spécifique est calculée à partir du débit Q2, de la pente et
    de la largeur maximale.

    Calcul de la pente et de la largeur à 10 fois la largeur du lit mineur (Slope_10, M_WIDTH_10) : Pour chaque transect
    une pente est calculée sur une distance de 10 fois la largeur du lit mineur, et une largeur moyenne est calculée
    sur cette même distance.

    Calcul de la puissance spécifique à 10 fois la largeur du lit mineur (PS_10) : La contrainte de cisaillement
    est recalculée avec les nouvelles valeurs de pente et de largeur.

    Prédiction de la granulométrie (D50, D84) : Des modèles de régression chargés depuis des fichiers pickle sont
    utilisés pour prédire le D50 et le D84 à partir de la contrainte de cisaillement PS_10.

Exportation et visualisation :

    Exportation des données : Les transects traités sont enregistrés dans un nouveau fichier shapefile.
    Création de graphiques de validation (optionnel) : Si make_validation_plots est activé, des graphiques
    comparant les valeurs originales et corrigées de l'accumulation des flux et de l'élévation sont générés
    pour chaque tributaire.

Journalisation :

    Un fichier log est mis à jour avec la date, l'heure, le nom de la rivière, la durée du traitement et les valeurs des
    constantes utilisées.

Auteur : Mathias Chabal - INRS
"""

from pathlib import Path
import pickle
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go

from neptune.tools import crs_river, get_columns_ln, get_col_dtype_ln, add_constant
from neptune.corrections import correct_z, correct_facc
from neptune.slope import calculate_local_slope, calculate_adaptive_slope_width
from neptune.network import HNetworkLinear
from neptune.logger import Logger

### Paramétres à modifier ###
river_directory = Path(r'D:\CPD') # dossier de travail
transects_trib_path = Path(r'D:\CPD\Lineaire\Transects_Ln_N1_CPD.shp') # transects linéaire N1
transects_principal_path = Path(r'D:\CPD\Transect_N2_CPD.shp') # transects principaux N2 surfacique
C_VAL_Q2 = (0.0000455899)  # Constante de c pour le calcul du débit Q2.
C_VAL_ORTHO = (0.0000021715) # Constante de c pour le calcul du débit Q_IMG (date qui couvrent le plus la rivière).
out_transects_path = Path(r'D:\CPD\Lineaire\Transects_Ln_N2_CPD.shp') # fichier de sortie
make_validation_plots = True # Créer des graphiques de validation
max_width_model_path = Path(r'D:\Supp_river\Modèles de prédiction\q2_width_predictions.pickle') # Modèle de prédiction

### Début du code ###
warnings.filterwarnings('ignore')
river_code = river_directory.name
log = Logger(river_code, river_directory, valeur_c_Q2=C_VAL_Q2, valeur_c_ortho=C_VAL_ORTHO,
             graphiques_validation=make_validation_plots)
CRS = crs_river(river_code)

print(f'Traitement de la rivière : {river_code}')

transects_trib = gpd.read_file(transects_trib_path)

transects_trib.rename(columns={"SEC_ID": "reach_ID"}, inplace=True)

n_nan = transects_trib['SUB_BV_ID'].isnull().sum()
if n_nan > 0:
    print(f'Avertissement: {n_nan} transects sans SUB_BV_ID — ils seront exclus.')
    transects_trib = transects_trib[transects_trib['SUB_BV_ID'].notna()].copy()

graph_path = Path(r"D:\CPD\Graphe\graphe_Ln_CPD.pkl")
graph = HNetworkLinear.from_file(graph_path)

transects_trib = transects_trib.sort_values(by=['SUB_BV_ID', 'Hierarchie', 'Nom', 'PK'],
                                            ascending=[True, True, True, True], ignore_index=True)

transects_trib.loc[transects_trib['LB_MIN'].isnull(), 'LB_MIN'] = np.nan
transects_trib.loc[transects_trib.FACC_max < 1_000, 'FACC_max'] = np.nan
n_trib = transects_trib['Nom'].nunique()

transects_principal_n2 = gpd.read_file(transects_principal_path)  # Pour Vérifier si le tributaire est accessible par
# rapport au chenal principal

if transects_principal_n2.crs.to_epsg() != CRS:
    transects_principal_n2.to_crs(CRS, inplace=True)

transects_trib["Accessible"] = True

for trib, sub_trib in tqdm(transects_trib.groupby("Nom"), total=n_trib, desc='Traitement des tributaires'):
    trib_index = sub_trib.index

    ref = sub_trib['FACC_max'].min()

    if np.isnan(ref):
        print(f'Le chenal {trib} a aucune valeurs valides dans la colonne FACC_COR.')

    facc = sub_trib['FACC_max'].copy()
    facc.iloc[-1] = ref
    facc_flipped = np.flip(facc.values)
    # Correction des valeurs de FACC
    corrected_facc = correct_facc(facc_flipped, ref)

    facc_interpolated = pd.Series(np.flip(corrected_facc), index=trib_index).interpolate().bfill().ffill()
    transects_trib.loc[trib_index, 'FACC_COR'] = facc_interpolated

    z = sub_trib['LB_MIN'].copy()
    z.replace(0, np.nan, inplace=True)  # Une valeur peut être égale à 0, évite problème dans la phase de correction
    z.iloc[0] = z.loc[z.notna()].min()  # Pour que la valeur la plus en aval soit la plus basse
    z = np.flip(z.values)
    ref = np.nanmax(z)

    if not np.isnan(ref) and not np.all(np.isnan(z)):
        REF = np.nanmax(z)
    else:
        print(f'Le chenal {trib} a aucune valeurs valides dans la colonne LB_min.')
        continue

    corrected_z = correct_z(z, ref)

    z_interpolated = pd.Series(np.flip(corrected_z), index=trib_index).interpolate().bfill().ffill()
    sub_trib['LB_MIN_COR'] = z_interpolated
    transects_trib.loc[trib_index, 'LB_MIN_COR'] = z_interpolated

    PK = sub_trib['PK'].copy()

    slope = calculate_local_slope(PK, z_interpolated)

    transects_trib.loc[trib_index, 'Pente'] = slope

    trib_hierarchy = sub_trib["Hierarchie"].iloc[0]

    if trib_hierarchy == 1:
        first_transect = sub_trib.loc[sub_trib.index == sub_trib["PK"].idxmin()][["reach_ID", "geometry"]]
        first_sec_id = first_transect["reach_ID"].iloc[0]

        closest = first_transect.sjoin_nearest(transects_principal_n2[["Accessible", "PK", "geometry"]],
                                               max_distance=50).reset_index(drop=True)

        if closest.shape[0] >= 1:
            idx = closest["PK"].idxmin()
            accessible = closest.at[idx, 'Accessible']

            if not accessible:
                transects_trib.loc[trib_index, 'Accessible'] = False
                tribs = graph.get_tributaries(first_sec_id)
                transects_trib.loc[transects_trib["Nom"].isin(tribs), "Accessible"] = False
                continue

    last_index = sub_trib.index.max()
    # Détermination de l'accessibilité
    for index, row in sub_trib.iterrows():
        if not row.Accessible:
            break

        curent_sec_id = row.reach_ID

        down_elev = row.LB_MIN_COR
        up_index = index + 20  # Équivaut à 100m de distance

        if up_index > last_index:
            continue

        up_elev = sub_trib.at[up_index, 'LB_MIN_COR']

        if (up_elev - down_elev) >= 12:
            transects_trib.loc[index:last_index, 'Accessible'] = False
            tribs = set(graph.get_tributaries(curent_sec_id))
            tribs.discard(trib)
            transects_trib.loc[transects_trib["Nom"].isin(tribs), "Accessible"] = False
            break


transects_trib['Q2'] = C_VAL_Q2 * transects_trib['FACC_COR'] ** 0.75
transects_trib['Q_IMG'] = C_VAL_ORTHO * transects_trib['FACC_COR'] ** 0.75
transects_trib['WIDTH'] = 2.5 * (transects_trib['FACC_COR'] / 10000000) ** 0.4

with open(max_width_model_path, 'rb') as f:
    width_prediction_model = pickle.load(f)

to_predict = add_constant(transects_trib['Q2'].astype(float).apply(np.log))
transects_trib['MAX_WIDTH'] = np.exp(width_prediction_model.predict(to_predict))

transects_trib['Pente'] = transects_trib['Pente'].abs()

# Calcule de la puissance spécifique
transects_trib['PS'] = 999.98 * 9.8 * transects_trib['Q2'] * transects_trib['Pente'] / transects_trib['MAX_WIDTH']

transects_trib['Slope_10'] = np.nan
transects_trib['M_WIDTH_10'] = np.nan
transects_trib = transects_trib.astype(get_col_dtype_ln(transects_trib.columns), errors='ignore')

for u, sub_trib in tqdm(transects_trib.groupby('Nom'), desc='Calcule de la pente avec le multiple de 10 : '):

    sub_trib['W_med'] = sub_trib['MAX_WIDTH'].rolling(7, center=True).median().bfill().ffill()
    trib_index = sub_trib.index

    pk = sub_trib['PK']
    elev = sub_trib['LB_MIN_COR']
    w_med = sub_trib['W_med']
    max_width = sub_trib['MAX_WIDTH']

    slope_10, m_width_10 = calculate_adaptive_slope_width(pk, elev, w_med, max_width)


    transects_trib.loc[trib_index, 'Pente_10'] = pd.Series(slope_10, index=trib_index).interpolate().bfill().ffill()
    transects_trib.loc[trib_index, 'M_WIDTH_10'] = pd.Series(m_width_10, index=trib_index).interpolate().bfill().ffill()

transects_trib['Pente_10'] = transects_trib['Pente_10'].abs()

transects_trib['PS_10'] = 999.98 * 9.8 * transects_trib['Q2'] * transects_trib['Pente_10'] / transects_trib[
    'M_WIDTH_10']

transects_trib['D50'] = 4.87 * (transects_trib['PS_10'] ** 0.63)
transects_trib['D84'] = 7.52 * (transects_trib['PS_10'] ** 0.68)


columns = transects_trib.columns
transects_trib = transects_trib[get_columns_ln(columns)].astype(get_col_dtype_ln(columns), errors='ignore')

transects_trib.to_file(out_transects_path, engine="pyogrio", use_arrow=True)
print(f'Fichier enregistrez sous : {out_transects_path}\n')

log.stop()

if make_validation_plots:
    validation_plots_dir = out_transects_path.parent.joinpath('Validation Plots')
    validation_plots_dir.mkdir(exist_ok=True)

    n = transects_trib['Nom'].nunique()

    for trib, sub_trib in tqdm(transects_trib.groupby("Nom"), desc='Creating validation plots', total=n):
        out_plot_dir = validation_plots_dir.joinpath(f'{trib}')
        out_plot_dir.mkdir(exist_ok=True)

        pk = sub_trib['PK']

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=pk, y=sub_trib['FACC_COR'], mode='lines', name='FACC corrigé'))
        fig1.add_trace(go.Scatter(x=pk, y=sub_trib['FACC_max'], mode='lines', name='FACC brut'))
        fig1.update_layout(title='Accumulation des flux', xaxis_title='PK', yaxis_title='Accumulation des flux (m2)')

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pk, y=sub_trib['LB_MIN_COR'], mode='lines', name='Elévation corrigée'))
        fig2.add_trace(go.Scatter(x=pk, y=sub_trib['LB_MIN'], mode='lines', name='Elévation brute'))
        fig2.update_layout(title='Profil en long', xaxis_title='PK', yaxis_title='Élévation (m)')

        fig1.write_html(f'{out_plot_dir}/{trib}_FACC_COR.html')
        fig2.write_html(f'{out_plot_dir}/{trib}_LB_MIN_COR.html')

log.print_log()

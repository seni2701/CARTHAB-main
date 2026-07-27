"""
Script pour ajouter les variables nécessaires pour le modèle de présence à des données qui ont déjà été traitées.
Pour la méthode surfacique.
"""

from pathlib import Path
import geopandas as gpd
from tqdm import tqdm

transects_path = Path(r'D:\STJSG\Transect_N2_STJSG.shp')
lower_order_sf_trib: str | None = None
order: int = 1
out_layer_path: Path | None = None

transects_ori = gpd.read_file(transects_path, engine="pyogrio", use_arrow=True)

transects_ori["distance"] = transects_ori["PK"]

if "Principal" not in transects_ori.columns:
    if "Principale" in transects_ori.columns:
        principal_col = "Principale"
    else:
        raise KeyError("La colonne 'Principal' n'existe pas dans le fichier des transects.")
else:
    principal_col = "Principal"

if any(transects_ori[principal_col] == "T"):
    transects_ori[principal_col] = transects_ori[principal_col].map({"T": True, "F": False})
else:
    transects_ori[principal_col] = transects_ori[principal_col].astype(bool)

transects = transects_ori[transects_ori[principal_col]].copy()
transects.sort_values("PK", inplace=True)
ori_index = transects.index
transects.reset_index(drop=True, inplace=True)

passe = "Passe" in transects.columns
prev_slopes = []
for r in tqdm(transects.itertuples(), total=len(transects), desc="Ajout de la variable max_slope_5"):
    i = r.Index

    if i == 0:
        transects.at[i, "slope_5"] = 0
        transects.at[i, "m_slope_5"] = 0
        continue

    if passe:
        if r.Passe == 1.:
            transects.at[i, "slope_5"] = 0
            transects.at[i, "m_slope_5"] = max(prev_slopes)
            continue

    current_elev = r.LB_Q25_COR
    previous_elev = transects.at[i - 1, "LB_Q25_COR"]

    slope_5 = (current_elev - previous_elev)/5

    prev_slopes.append(slope_5)

    transects.at[i, "slope_5"] = slope_5
    transects.at[i, "m_slope_5"] = max(prev_slopes)

transects.set_index(ori_index, inplace=True)

transects_ori.loc[transects.index, "slope_5"] = transects["slope_5"]
transects_ori.loc[transects.index, "m_slope_5"] = transects["m_slope_5"]

if lower_order_sf_trib is not None and Path(lower_order_sf_trib).exists():
    lower_trib_gdf = gpd.read_file(lower_order_sf_trib, engine="pyogrio", use_arrow=True)

    downstream_point = transects_ori.geometry.iloc[0].centroid
    idx = lower_trib_gdf.distance(downstream_point).idxmin()

    if "distance" in lower_trib_gdf.columns and "m_slope_5" in lower_trib_gdf.columns:
        if order > 1:
            lower_trib_dist = lower_trib_gdf["distance"].loc[idx]
            transects_ori["distance"] = transects_ori["distance"] + lower_trib_dist

        lower_trib_max_slope = lower_trib_gdf["m_slope_5"].loc[idx]
        transects_ori.loc[transects_ori["m_slope_5"] < lower_trib_max_slope, "m_slope_5"] = lower_trib_max_slope

    else:
        raise KeyError("Les colonnes 'distance' et/ou 'm_slope_5' n'existent pas dans le fichier du tributaire d'ordre "
                       "inférieur, rouler ce script pour le tributaire surfacique d'ordre inférieur avant celui-ci.")

if out_layer_path is not None:
    transects_ori.to_file(out_layer_path, engine="pyogrio")
else:
    transects_ori.to_file(transects_path, engine="pyogrio")

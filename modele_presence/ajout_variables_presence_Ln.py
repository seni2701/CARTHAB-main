"""
Script pour ajouter les variables nécessaires pour le modèle de présence à des données qui ont déjà été traitées.
Pour la méthode linéaire
"""

from pathlib import Path
import geopandas as gpd
from tqdm import tqdm
from neptune.network import HNetworkLinear

transects_path = Path(r'D:\STJSG\Lineaire\Transects_Ln_N2_STJSG.shp')
out_layer_path: str | None  = None
graph_path = Path(r"D:\STJSG\Graphe\graphe_Ln_STJSG.pkl")
all_sf_rivers_path = Path(r'D:\STJSG\Transect_N2_STJSG.shp')

transects = gpd.read_file(transects_path, engine="pyogrio", use_arrow=True)
# Trie les valeurs pour que les tributaires avec une hiérarchie de 1 soient traités en premier et pour que le traitement
# se fasse de l'aval vers l'amont.
transects.sort_values(by=["Hierarchie", "Nom", "PK"], inplace=True, ignore_index=True)

network = HNetworkLinear.from_file(graph_path)
n_trib = transects.Nom.nunique()

sf_trib_gdf = gpd.read_file(all_sf_rivers_path, engine="pyogrio", use_arrow=True)

if "distance" not in sf_trib_gdf.columns or "m_slope_5" not in sf_trib_gdf.columns:
    raise KeyError("Les colonnes 'distance' et/ou 'm_slope_5' n'existent pas dans le fichier des rivières surfaciques")

for trib, df in tqdm(transects.groupby("Nom", sort=False), total=n_trib, desc="Ajout de la variable max_slope_5"):

    ori_index = df.index
    df.reset_index(drop=True, inplace=True)

    hierarchie = df["Hierarchie"].iloc[0]
    
    if hierarchie == 0:
        downstream_point = df.geometry.iloc[0].centroid
        idx = sf_trib_gdf.distance(downstream_point).idxmin()
        down_pk = sf_trib_gdf.loc[sf_trib_gdf.index == idx, "PK"].iloc[0]
        df["PK"] = df["PK"] + down_pk

    prev_slopes = []
    prev_reach_id = None
    prev_down_transects = None
    downstream_transects_empty = False
    for r in df.itertuples():
        i = r.Index
        current_reach_id = r.reach_ID
        current_elev = r.LB_MIN_COR

        # Éviter de refaire une recherche si le reach_id précédent est le même, car sec_ids ne changent pas dans ce cas
        if prev_reach_id != current_reach_id:
            sec_ids = network.get_reach_ids(r.reach_ID, upstream=False)
            down_transects = transects[transects.reach_ID.isin(sec_ids)]

            # Gére les cas où il n'y a pas de transects en aval avec la méthode linéaire mais en surfacique
            if down_transects.empty and sec_ids:
                downstream_transects_empty = True
                downstream_point = r.geometry.centroid
                idx = sf_trib_gdf.distance(downstream_point).idxmin()

                # Ne récupère que le transect le plus proche
                down_transects = sf_trib_gdf[sf_trib_gdf.index == idx]

        else:
            down_transects = prev_down_transects

        # Gére le cas des premiers transects
        if i == 0:
            if hierarchie <= 1:
                # Pas possible de calculer les variables slope_5 et max_slope_5 pour les premiers transects.
                df.at[i, "slope_5"] = 0
                df.at[i, "m_slope_5"] = 0
                df.at[i, "distance"] = r.PK
            else:
                # Gère le cas où il faut aller chercher les valeurs sur la partie surfacique.
                elev_col = "LB_Q25_COR" if "LB_Q25_COR" in down_transects.columns else ("LB_Q25" if "LB_Q25" in down_transects.columns else "LB_MIN_COR")
                if downstream_transects_empty:
                    previous_elev = down_transects[elev_col].iloc[0]
                    max_slope_5 = down_transects["m_slope_5"].iloc[0]
                else:
                    if "Hierarchie" in down_transects.columns:
                        filtered = down_transects[down_transects["Hierarchie"] == hierarchie-1]["LB_MIN_COR"]
                        previous_elev = filtered.iloc[-1] if len(filtered) > 0 else (down_transects[elev_col].iloc[0] if len(down_transects) > 0 else current_elev)
                    else:
                        previous_elev = down_transects[elev_col].iloc[0]
                    max_slope_5 = down_transects["slope_5"].max()

                slope_5 = (current_elev - previous_elev) / 5
                dist = down_transects["distance"].max() + r.PK

                df.at[i, "slope_5"] = slope_5
                df.at[i, "m_slope_5"] = max_slope_5
                df.at[i, "distance"] = dist

                prev_slopes.append(slope_5)
                prev_slopes.append(max_slope_5)

            continue

        previous_elev = df.at[i - 1, "LB_MIN_COR"]

        slope_5 = (current_elev - previous_elev)/5

        prev_slopes.append(slope_5)

        if hierarchie <= 1:
            max_slope_5 = max(prev_slopes)
            dist = r.PK

        else:
            # Gère le cas où il faut aller chercher les valeurs sur la partie surfacique. Si surfacique, on cherche
            # max_slope_5 car, down_transects ne représente qu'une entité et non tout les transects en aval.
            if downstream_transects_empty:
                slope_col = "m_slope_5"
            else:
                slope_col = "slope_5"

            max_slope_5 = max(down_transects[slope_col].tolist() + prev_slopes)
            dist = down_transects["distance"].max() + r.PK

        df.at[i, "slope_5"] = slope_5
        df.at[i, "m_slope_5"] = max_slope_5
        df.at[i, "distance"] = dist

        prev_reach_id = current_reach_id
        prev_down_transects = down_transects

    df.set_index(ori_index, inplace=True)

    transects.loc[df.index, "slope_5"] = df["slope_5"]
    transects.loc[df.index, "m_slope_5"] = df["m_slope_5"]
    transects.loc[df.index, "distance"] = df["distance"]

if out_layer_path is not None:
    transects.to_file(out_layer_path, engine="pyogrio")
else:
    transects.to_file(transects_path, engine="pyogrio")

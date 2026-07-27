import arcpy as ap
from pathlib import Path
import geopandas as gpd

# Chemins fixes
index_path = Path(r"D:\MLBG\PHOTOS\Index\Index_photos_Malbaie11_nettoye_nettoye.shp")
out_index_path = r"D:\MLBG\PHOTOS\Index\Index_photos_Malbaie11_utilise.shp"

# Connexion au projet ArcGIS
aprx = ap.mp.ArcGISProject(r"C:\Users\snabr\Documents\ArcGIS\Projects\MLBG\MLBG.aprx")
map_obj = aprx.listMaps()[0]

# Récupérer le groupe Principale_rive
group_layer = next(l for l in map_obj.listLayers() if l.isGroupLayer and l.name == "MLB")

# Ordre réel des couches (haut = priorité basse)
ordered_layers = [
    "q16049_257_ortho.tif",
    "q16049_272_ortho.tif",
    "q16049_273_ortho.tif",
    "q16049_274_ortho.tif",
    "q16049_384_ortho.tif",
    "q16049_383_ortho.tif",
    "q16049_381_ortho.tif",
    "q16049_379_ortho.tif",
    "q16049_377_ortho.tif",
]

index = gpd.read_file(index_path)
true_file_name = index["FULL_PATH"].apply(lambda x: Path(x).name)
index = index[true_file_name.isin(ordered_layers)]

# Hiérarchie : couche du haut = 1, couche du bas = total
for i, layer_name in enumerate(ordered_layers):
    hierarchie = i + 1  # 104 en haut → 1, 117 en bas → 16
    index.loc[true_file_name == layer_name, 'Hierarchie'] = hierarchie

index.sort_values(by='Hierarchie', ascending=True, inplace=True, ignore_index=True)
index.to_file(out_index_path, index=False)

print("Fichier créé avec succès !")
print(index[["NOM_IMAGE", "Hierarchie"]])
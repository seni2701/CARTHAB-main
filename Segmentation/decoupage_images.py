import os
import tempfile
import shutil
import arcpy as ap
from pathlib import Path
import geopandas as gpd
import numpy
import rasterio
from rasterio.enums import Resampling as Resampling_enums
from rasterio.warp import reproject
from rasterio.warp import Resampling as Resampling_warp

# Récupérer les paramètres de l'outil
if ap.GetParameter(0):
    index_path = ap.GetParameter(0)
    ortho_dir = Path(ap.GetParameterAsText(1))
else:
    index_path = Path(r"D:\MLBG\PHOTOS\Index\Index_photos_Malbaie11_utilise.shp")
    ortho_dir = Path(r"D:\MLBG\PHOTOS")

try:
    index_path = Path(index_path.dataSource)
except AttributeError:
    index_path = Path(str(index_path))

if not index_path.suffix:
    index_path = index_path.with_suffix(".shp")

index = gpd.read_file(index_path)
index.sort_values(by='Hierarchie', ascending=False, inplace=True, ignore_index=True)

count = index.shape[0]
ap.AddMessage(f"Nombre de fichier à découper : {count}")
ap.SetProgressor("step", "Découpage des orthos", 0, count, 1)

sindex = index.sindex
left_index, right_index = sindex.query(index.geometry, predicate='intersects')
exact_mask = left_index != right_index  # évite auto-intersection
left_index, right_index = left_index[exact_mask], right_index[exact_mask]

for left, right in zip(left_index, right_index):
    left_hierarchy = index.at[left, "Hierarchie"]
    right_hierarchy = index.at[right, "Hierarchie"]

    # Changer cette condition dans le script de découpage :
    if left_hierarchy > right_hierarchy:
        underlying_image_fp = index.at[left, 'FULL_PATH']   # hiérarchie haute = dessous
        upper_image_fp = index.at[right, 'FULL_PATH']        # hiérarchie basse = dessus

        # Ouvrir les deux images pour construire le masque
        image_under = rasterio.open(underlying_image_fp)
        image_over = rasterio.open(upper_image_fp)

        # dataset_mask() retourne 255=valide, 0=nodata — indépendant des valeurs de pixels
        data_over = image_over.read()
        image_over_mask = numpy.any(data_over != 0, axis=0).astype(numpy.uint8) * 255

        image_over_mask_reprojected = numpy.zeros(
            (image_under.height, image_under.width), dtype=numpy.uint8
        )
        reproject(
            source=image_over_mask,
            destination=image_over_mask_reprojected,
            src_transform=image_over.transform,
            dst_transform=image_under.transform,
            src_crs=image_over.crs,
            dst_crs=image_under.crs,
            resampling=Resampling_warp.nearest,
        )

        # Pixels de l'image du dessous couverts par l'image du dessus
        image_under_bad_mask = image_over_mask_reprojected == 255

        # Lire les données et le profil AVANT de fermer
        data = image_under.read()
        profile = image_under.profile.copy()

        image_under.close()
        image_over.close()

        # Modifier les données en mémoire
        data[:, image_under_bad_mask] = 0

        # Mettre à jour le profil pour supporter les overviews (tiled requis)
        profile.update(
            nodata=0,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
        )

        # Écrire dans un fichier temporaire (dossier temp système, pas de verrou ArcGIS)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif")
        os.close(tmp_fd)

        try:
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(data)
                overview_levels = [2, 4, 8, 16]
                dst.build_overviews(overview_levels, Resampling_enums.nearest)
                dst.update_tags(ns="rio_overview", resampling="nearest")

            # Copier vers la destination (écrase l'original sans rename cross-device)
            shutil.copy2(tmp_path, underlying_image_fp)

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        ap.SetProgressorPosition()

ap.AddMessage("Terminé")
print("Terminé")
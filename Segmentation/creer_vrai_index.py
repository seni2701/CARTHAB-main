import tempfile
import arcpy as ap
from pathlib import Path
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.ops import transform

def transform_coordinates(geom, from_epsg, to_epsg):
    transformer = Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    return transform(transformer.transform, geom)

# Chemins fixes
index_path = Path(r"D:\MLBG\PHOTOS\Index\Index_photos_Malbaie11_utilise.shp")
out_file_path = r"D:\MLBG\PHOTOS\Index\index_ortho_MLBG_reel.shp"

# Connexion au projet ArcGIS
aprx = ap.mp.ArcGISProject(r"C:\Users\snabr\Documents\ArcGIS\Projects\MLBG\MLBG.aprx")

index = gpd.read_file(index_path)
full_name = index["FULL_PATH"]
index_copy = index.copy()
index_epsg_code = 2947

files = full_name.apply(Path)
n_files = len(files)

ap.SetProgressor("step", "Récréation de l'index", 0, n_files, 1)
ap.AddMessage(f"Traitement de {n_files} fichier(s)")

for i, file in enumerate(files):
    ap.AddMessage(f"Traitement du fichier {i+1}/{n_files} : {file.name}")
    raster = ap.Raster(str(file.joinpath('Band_1')))
    raster_epsg_code = raster.spatialReference.factoryCode
    classified = ap.sa.Con(raster > 0, 1, 0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        temp_file = Path(tmpdir).joinpath(f'{file.stem}.shp')
        ap.RasterToPolygon_conversion(classified, str(temp_file), simplify='NO_SIMPLIFY')
        shapes = gpd.read_file(temp_file)
        shapes = shapes.set_crs(2947, allow_override=True)
        mask = shapes.area > 50
        geom = shapes[mask].unary_union
        index_copy.loc[full_name == str(file), 'geometry'] = geom
    ap.SetProgressorPosition()

index_copy.to_file(out_file_path, index=False)
print("Fichier créé avec succès !")
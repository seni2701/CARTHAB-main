import sys
sys.path.insert(0, r"D:\neptune")

import arcpy as ap
from pathlib import Path
import pandas as pd
from neptune.tools import crs_river
import geopandas as gpd
from shutil import rmtree

ap.CheckOutExtension("3D")
ap.CheckOutExtension("Spatial")
ap.CheckOutExtension("ImageAnalyst")
    
wd_path = Path(r"D:\MLBG")
index_path = Path(r"D:\MLBG\PHOTOS\Index\index_ortho_MLBG_reel.shp")
centerline_fp = Path(r"D:\MLBG\Ligne_MLBG.shp")
buffer_size = int(200)
roads_file = Path(r"D:\Reseau de transport\reseaux_trasnport.shp")
out_water_fp = Path(r"D:\MLBG\eau_mlb.shp")
out_active_fp = Path(r"D:\MLBG\active_mlb.shp")
remove_temp_DB = True

ap.env.overwriteOutput = True

gdb_path = Path(wd_path) / 'temp_db.gdb'
ap.management.CreateFileGDB(str(wd_path), "temp_db")

ap.env.workspace = str(gdb_path)

CRS = 2947
TRUE_VALUE = 1
FALSE_VALUE = 0

try:
    fp = Path(index_path.dataSource).with_suffix('.shp')
    index = gpd.read_file(fp)
except AttributeError as e:
    index = gpd.read_file(index_path)

# Détection robuste de la colonne contenant les chemins vers les rasters
_PATH_KEYWORDS = ('full_path', 'path', 'chemin', 'filepath', 'raster', 'image', 'filename')
_path_col = next(
    (c for c in index.columns if c.lower() in _PATH_KEYWORDS),
    next((c for c in index.columns if any(kw in c.lower() for kw in _PATH_KEYWORDS)), None)
)
if _path_col is None:
    raise KeyError(
        f"Aucune colonne de chemin trouvée dans l'index. "
        f"Colonnes disponibles : {list(index.columns)}. "
        f"Renommez la colonne contenant les chemins vers les rasters en 'FULL_PATH'."
    )
ap.AddMessage(f"Colonne de chemin utilisée : '{_path_col}'")
raster_files = index[_path_col].apply(Path).to_list()
seg_files = []
mk_files = []

ap.SetProgressor("step", "Segmentation des orthos", 0, len(raster_files), 1)
counter = 1
for file in raster_files:
    ap.AddMessage(f"Traitement du fichier {counter}/{len(raster_files)} : {file}")
    raster = ap.Raster(str(file))

    buffer = f'temp_buffer_{buffer_size}'
    ap.analysis.Buffer(str(centerline_fp), buffer, buffer_size, dissolve_option='ALL')

    buffer_proj = f'temp_buffer_proj_{buffer_size}'
    sr_mtm5 = ap.SpatialReference(2947)
    ap.management.Project(buffer, buffer_proj, sr_mtm5)
    clipped_raster = 'clipped_raster'
    ap.management.Clip(raster, out_raster=clipped_raster, in_template_dataset=buffer_proj, nodata_value="0",
                    clipping_geometry="ClippingGeometry")

    nir = 'nir_temp' + file.stem
    ap.management.MakeRasterLayer(in_raster=clipped_raster, out_rasterlayer=nir, band_index=[4])
    nir_raster = ap.Raster(nir)

    red = 'temp_red'
    ap.management.MakeRasterLayer(in_raster=clipped_raster, out_rasterlayer=red, band_index=[1])
    red_raster = ap.Raster(red)

    threshold_fp = file.with_suffix('.csv')
    threshold_df = pd.read_csv(threshold_fp, sep=',')
    ndvi_threshold, nir_threshold = threshold_df[['ndvi', 'nir']].iloc[0]

    ndvi = ap.ia.NDVI(clipped_raster, nir_band_id=4, red_band_id=1)

    ndvi_reclass = ap.sa.Con(ndvi < ndvi_threshold, TRUE_VALUE, FALSE_VALUE)
    nir_reclass = ap.sa.Con((nir_raster < nir_threshold) & (ndvi < ndvi_threshold), TRUE_VALUE, FALSE_VALUE)

    nir_ndvi = ap.sa.Plus(nir_reclass, ndvi_reclass)

    valid_mask = ap.sa.Con(red_raster > 0, TRUE_VALUE, FALSE_VALUE)
    valid_seg = ap.sa.ExtractByMask(nir_ndvi, valid_mask, extraction_area='INSIDE')

    filtered_seg = ap.sa.MajorityFilter(valid_seg, number_neighbors='EIGHT', majority_definition='MAJORITY')
    cleaned_seg = ap.sa.BoundaryClean(filtered_seg, sort_type='NO_SORT', number_of_runs='TWO_WAY')

    seg_fp = 'SEG_' + file.stem
    seg_files.append(seg_fp)
    ap.conversion.RasterToPolygon(cleaned_seg, seg_fp, simplify='SIMPLIFY')

    counter += 1
    ap.SetProgressorPosition()


merged_files = 'merged_seg'
ap.management.Merge(seg_files, merged_files, add_source='NO_SOURCE_INFO')
ap.AddMessage("Fichiers combinés")

diss_file_fp = 'dissolved_file'
ap.analysis.PairwiseDissolve(merged_files, diss_file_fp, dissolve_field='gridcode', multi_part='SINGLE_PART')
ap.AddMessage("Fichiers fusionnés")

diss_file = gpd.read_file(gdb_path, layer=diss_file_fp)

if diss_file.crs.to_epsg() != CRS:
    diss_file.to_crs(CRS, inplace=True)

water = diss_file[diss_file['gridcode'] == 2]
water = water.explode()
water.reset_index(drop=True, inplace=True)
water["area"] = water.area

active = diss_file[diss_file['gridcode'] != 0]
active = active.dissolve().explode()
active.reset_index(drop=True, inplace=True)
active["area"] = active.area

try:
    fp = Path(centerline_fp.dataSource).with_suffix('.shp')
    centerline = gpd.read_file(fp)
except AttributeError as e:
    centerline = gpd.read_file(centerline_fp)

if centerline.crs.to_epsg() != CRS:
    centerline.to_crs(CRS, inplace=True)

centerline_bf = centerline.unary_union

intersect_result_water = water.sindex.query(centerline_bf, 'intersects')
water = water.loc[intersect_result_water]
intersect_result_bank = active.sindex.query(centerline_bf, 'intersects')
active = active.loc[intersect_result_bank]

try:
    fp = Path(roads_file.dataSource).with_suffix('.shp')
    roads = gpd.read_file(fp)
except AttributeError as e:
    roads = gpd.read_file(roads_file)

roads.to_crs(CRS, inplace=True)
intersect_result_bank_2 = active.sindex.query(roads.union_all(), 'intersects')
active = active[~active.index.isin(intersect_result_bank_2)]

water.to_file(out_water_fp, index=False)
active.to_file(out_active_fp, index=False)

if remove_temp_DB:
    rmtree(gdb_path, ignore_errors=True)
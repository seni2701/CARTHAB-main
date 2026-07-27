import shutil as ut
from tqdm import tqdm
from pathlib import Path
import geopandas as gpd
import arcpy as ap

index_fp = ap.GetParameter(0)
ligne_fp = ap.GetParameter(1)
src_directory = Path(ap.GetParameterAsText(2))
dest_directory = Path(ap.GetParameterAsText(3))

try:
    index_fp = Path(index_fp.dataSource)

except AttributeError as e:
    index_fp = Path(str(index_fp))

try:
    ligne_fp = Path(ligne_fp.dataSource)

except AttributeError as e:
    ligne_fp = Path(str(ligne_fp))

ligne = gpd.read_file(ligne_fp)
index = gpd.read_file(index_fp)

if ligne.crs.to_epsg() != index.crs.to_epsg():
    ligne.to_crs(index.crs, inplace=True)

file_list = index[index.intersects(ligne.union_all().buffer(15))]['File'].tolist()
dest_directory.mkdir(exist_ok=True)

for file in tqdm(file_list):
    if (src_directory / file).exists():
        ut.move(src_directory / file, dest_directory)

    else:
        print(f"File not found : {file}")

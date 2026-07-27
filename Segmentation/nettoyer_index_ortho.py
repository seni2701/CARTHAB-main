import arcpy as ap
from pathlib import Path
import geopandas as gpd
import pandas as pd

index_layer = ap.GetParameter(0)
field_name = ap.GetParameterAsText(1)
ortho_folder = Path(ap.GetParameterAsText(2)).resolve()
fields_to_keep = ap.GetParameterAsText(3)

try:
    index_path = Path(index_layer.dataSource)
except AttributeError as e:
    index_path = Path(str(index_layer))

index = gpd.read_file(index_path)

# Ne garder que les 10 premiers caractères du nom de fichier pour pouvoir avoir une base de comparaison
index['short_name'] = index[field_name].str[:10]
index['short_name'] = index['short_name'].str.lower()

photo_list = []
for path in ortho_folder.glob('*.tif'):
    photo_list.append(path.stem)

# Extraire la base de comparaison pour les noms de fichier réel
s = pd.Series(photo_list).str.lower()
s_short = s.str[:10]

if 'COULEUR' not in index.columns:
    mask = index[field_name].str.lower().str.contains('nir')
    index = index[~mask]
else:
    mask = index['COULEUR'] != 'IRP'
    index = index[mask]

# Initialiser FULL_PATH à None pour garantir que la colonne existe toujours
index['FULL_PATH'] = None

index_copy = index.copy()
for idx, row in index_copy.iterrows():
    short_name = row['short_name']
    mask = s_short == short_name
    if mask.any():
        real_name = s[mask].iloc[0]
        index.loc[idx, 'NOM_IMAGE'] = real_name
        index.loc[idx, 'FULL_PATH'] = str(ortho_folder.joinpath(real_name + '.tif'))

index = index[~index['NOM_IMAGE'].isna()]
index = index[~index['FULL_PATH'].isna()]

cols_to_keep = fields_to_keep.split(';')
if 'DATE_PHOTO' not in cols_to_keep and 'DATE_PHOTO' in index.columns:
    cols_to_keep = ['NOM_IMAGE', 'DATE_PHOTO'] + cols_to_keep + ['FULL_PATH', 'geometry']
else:
    cols_to_keep = ['NOM_IMAGE'] + cols_to_keep + ['FULL_PATH', 'geometry']

# Dédupliquer en préservant l'ordre
seen = set()
cols_to_keep = [c for c in cols_to_keep if not (c in seen or seen.add(c))]

index = index[cols_to_keep]

file_name = index_path.stem
out_index_path = index_path.parent.joinpath(f'{file_name}_nettoye.shp')
index.to_file(out_index_path)
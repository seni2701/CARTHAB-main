import geopandas as gpd

gdf = gpd.read_file(r'D:\MLBG\Transects_Ln_N1_NVL.shp', engine='pyogrio')
print('Total transects:', len(gdf))
print('NaN dans SUB_BV_ID:', gdf['SUB_BV_ID'].isnull().sum())
print('Dtype SUB_BV_ID:', gdf['SUB_BV_ID'].dtype)
print('Valeurs uniques:', gdf['SUB_BV_ID'].unique())

# Ajoutez dans check.py
nan_rows = gdf[gdf['SUB_BV_ID'].isnull()]
print('PK min/max des NaN:', nan_rows['PK'].min(), '/', nan_rows['PK'].max())
print('Nom des NaN:', nan_rows['Nom'].unique())
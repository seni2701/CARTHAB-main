import geopandas as gpd
import requests
from urllib.parse import urljoin
from pathlib import Path
import arcpy as ap
from arcpy.ia import Merge

### Parameters to modify ###
index_file = ap.GetParameter(0)
river_dir = Path(ap.GetParameterAsText(1))  # destination dir

### Start of the code ###
count = int(ap.GetCount_management(index_file)[0])
ap.SetProgressor("step", "Téléchargement des MNT", 0, count, 1)
with ap.da.SearchCursor(index_file, ["lidar_url"]) as cursor:
    for i, row in enumerate(cursor):
        ap.SetProgressorPosition()
        root_url = row[0]
        zone = root_url.split('/')[-2]

        url = urljoin(root_url, f"MNT_{zone}.tif")
        destination_suffix = url.split('_')[0]
        destination_path = river_dir.joinpath(f"MNT_{zone}.tif")

        if destination_path.exists():
            ap.AddMessage(f"Fichier existe déjà : {destination_path}")
            ap.SetProgressorPosition()
            continue

        response = requests.get(url)
        with open(destination_path, "wb") as file:
            file.write(response.content)
            
        ap.SetProgressorPosition()
        ap.AddMessage(f"Fichier téléchargé {i + 1}/{count} : {destination_path}")

ap.SetProgressorPosition(count)
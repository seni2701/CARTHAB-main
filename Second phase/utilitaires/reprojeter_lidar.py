import pdal
from pathlib import Path
import laspy as lp
from neptune.tools import crs_river
import json

dir_lidar = Path(r'')
river_crs = crs_river('')  # Ajouter l'acronyme de votre rivière


files = dir_lidar.glob('*.la*')

for file in files:
    crs = lp.open(file).header.crs.to_epsg()

    if crs != river_crs:
        pipeline = {
            'pipeline': [
                {
                    "type": "readers.las",
                    "filename": str(file)
                },
                {
                    "type": "filters.reprojection",
                    "in_srs": f"EPSG:{crs}",
                    "out_srs": f"EPSG:{river_crs}"
                },
                {
                    "type": "writers.las",
                    "filename": str(file)
                }
            ]
        }

        pdal.Pipeline(json.dumps(pipeline)).execute()

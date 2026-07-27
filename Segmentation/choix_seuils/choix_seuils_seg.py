from pathlib import Path
from typing import Annotated
import gc
import json
from warnings import filterwarnings
import numpy as np
import geopandas as gpd
import rasterio as rio
import pandas as pd
from src.napari_visualisation import NapariVisualisation
import typer as tp
from rich.progress import track
from rich import print
from rich.table import Table
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError

filterwarnings("ignore", category=RuntimeWarning)

# Initial thresholds to start the calibration
NIR_THRESHOLD = 90
NDVI_THRESHOLD = 0.

BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "last_image.json"

class PathValidation(Validator):
    def validate(self, doc):
        path = doc.text

        if not Path(path).exists():
            raise ValidationError(message=f"Le chemin '{path}' n'existe pas.",
                                  cursor_position=len(doc.text))

        if not Path(path).is_file():
            raise ValidationError(message=f"Le chemin '{path}' n'est pas un fichier.",
                                  cursor_position=len(doc.text))

# Style pour le prompt
prompt_style = Style.from_dict({
    'prompt': '#f6ed78 bold italic',
})


def ask_path(message: str) -> str:
    """Demande un chemin avec autocomplétion."""
    completer = PathCompleter(expanduser=True)
    return pt_prompt(
        [('class:prompt', message), ('', ': ')],
        completer=completer,
        style=prompt_style,
        complete_while_typing=True,
        validator=PathValidation(),
        validate_while_typing=False
    )


app = tp.Typer()


def validate_paths(ctx: tp.Context, param: tp.CallbackParam, value: str | None):
    """Callback pour valider que les chemins sont fournis si --last-file n'est pas utilisé."""
    # Si la valeur est déjà fournie, on la retourne
    if value is not None:
        return value

    # Récupérer les autres paramètres déjà parsés
    last_file = ctx.params.get("last_file", False)

    # Si --last-file est activé, on peut récupérer les valeurs du fichier metadata
    if last_file and METADATA_PATH.exists():
        metadata = get_last_file_metadata()
        if param.name == "index_path":
            return metadata.get("index_path")
        elif param.name == "center_line_path":
            return metadata.get("line_path")

    # Sinon, on demande la valeur à l'utilisateur avec autocomplétion
    if param.name == "index_path":
        return ask_path("Index des orthos ")
    elif param.name == "center_line_path":
        return ask_path("Ligne centrale ")

    return value


def save_curent_file_metadata(index_path, file_index, file_path, line_path):
    file_metadata = {"index_path": str(index_path),
                     "line_path": str(line_path),
                     "file_index": file_index,
                     "file_path": str(file_path),
                     "file_name": file_path.stem}

    with open(METADATA_PATH, "w") as f:
        json.dump(file_metadata, f, indent=4)


def get_last_file_metadata():
    with open(METADATA_PATH, "r") as f:
        content = f.read()

    if not content.strip():
        return None

    return json.loads(content)


def list_tifs(path: str):
    gdf = gpd.read_file(path)
    gdf.sort_values(by='Hierarchie', inplace=True, ignore_index=True)
    files_list = gdf["FULL_PATH"].tolist()
    return files_list


def export_file(file_path: Path, theresholds: tuple):

    file_output_path = file_path.with_suffix(".csv")
    data = {
        "image_name": file_path.stem,
        "ndvi": theresholds[0],
        "nir": theresholds[1]
    }

    df = pd.DataFrame(data=data, index=[0])
    df = df.set_index("image_name")
    df.to_csv(file_output_path, sep=",")


def print_table(data: list[list[str]]):

    table = Table(title="\nSeuils enregistrés :", show_header=True, title_justify="left", header_style="bold blue")
    table.add_column("Image", justify="left")
    table.add_column("NDVI", justify="center")
    table.add_column("NIR", justify="center")

    for row in data:
        table.add_row(*row)

    print(table)


@app.command()
def main(last_file: Annotated[bool, tp.Option("--last-file", "-l", is_eager=True)] = False,
         index_path: Annotated[str | None, tp.Option(callback=validate_paths)] = None,
         center_line_path: Annotated[str | None, tp.Option(callback=validate_paths)] = None):

    metadata = get_last_file_metadata()

    # Set the paths to the photos
    if index_path is None:
        index_path = metadata["index_path"]

    if center_line_path is None:
        center_line_path = metadata["line_path"]

    files_list = list_tifs(index_path)

    if last_file:
            last_file_path = Path(metadata["file_path"])
            last_file_index = metadata["file_index"]

            if last_file_path in files_list:
                files_list = files_list[last_file_index:]

            else:
                print("[bold red]The last file is not in the list of photos.[/bold red]")

    visualisation = NapariVisualisation(center_line_path)
    threshold_data = []
    try:
        for i, file in track(enumerate(files_list), total=len(files_list)):
            file = Path(file)

            save_curent_file_metadata(index_path, i, file, center_line_path)
            with rio.open(file) as src:
                visualisation.dataset_rio = src
                extent_clip = visualisation.intersect_tif_and_shp()
                
                # Determine window and target shape for visualization
                height_orig = extent_clip.loc[2, "Index"] - extent_clip.loc[0, "Index"]
                width_orig = extent_clip.loc[3, "Index"] - extent_clip.loc[1, "Index"]

                out_shape = (height_orig, width_orig)

                # Clip bands using windowed read
                window = rio.windows.Window(extent_clip.loc[1, "Index"], extent_clip.loc[0, "Index"],
                                           width_orig, height_orig)

                red: np.ndarray = src.read(1, window=window)
                nir: np.ndarray = src.read(4, window=window)
                
                # Convert to float once for NDVI calculation
                red_f = red.astype(float)
                nir_f = nir.astype(float)
                
                ndvi = (nir_f - red_f) / (nir_f + red_f)
                ndvi[np.isnan(ndvi)] = np.nan
                
                rgb = visualisation.get_RGB(src, extent_clip, out_shape=out_shape)
                
                # Initial mask calculation is now handled inside calibrate_threshold for clarity
                # but we keep the same initial values
                result = visualisation.calibrate_threshold(rgb, 
                                                           ndvi, nir,
                                                           NIR_THRESHOLD,
                                                           NDVI_THRESHOLD)
                
                # Check if we should quit the program
                if result is None: 
                    break
                    
                ndvi_threshold, nir_threshold, quit_program = result

                threshold_data.append([file.stem, str(ndvi_threshold), str(nir_threshold)])

                if quit_program:
                    print("[bold green]Fin du programme demandée par l'utilisateur.[/bold green]")
                    break

                export_file(file, (ndvi_threshold, nir_threshold))
                    
                # Libérer explicitement la mémoire des arrays numpy
                del red, nir, red_f, nir_f, ndvi, rgb
                gc.collect()

        print_table(threshold_data)

    finally:
        if visualisation.viewer is not None:
            try:
                visualisation.viewer.close()

            except RuntimeError:
                # Viewer already closed or partially deleted
                pass


if __name__ == '__main__':
    app()

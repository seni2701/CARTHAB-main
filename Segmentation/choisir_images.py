import arcpy
from pathlib import Path

# Récupérer les paramètres de l'outil
input_layer = arcpy.GetParameter(0)

aprx = arcpy.mp.ArcGISProject("CURRENT")
active_map = aprx.activeMap

layers_set = set()
for layer in active_map.listLayers():
    layers_set.add(layer.name)

try:
    # Le curseur ne lira que les entités sélectionnées si une sélection existe sur input_layer
    with arcpy.da.SearchCursor(input_layer, ["FULL_PATH"]) as cursor:
        for row in cursor:
            file_info = row[0] # Nom ou chemin partiel du fichier depuis le champ

            if not file_info:
                arcpy.AddWarning(f"Valeur vide trouvée dans le champ 'FUL_PATH' pour une entité sélectionnée.")
                continue

        # Vérifier si le chemin est absolu ou relatif, c-a-d quel champ de l'index est utilisé (FULL_PATH ou NOM_IMAGE).

            full_path = Path(file_info)

            arcpy.AddMessage(f"Traitement du fichier : {full_path}")

            if arcpy.Exists(full_path):
                try:
                    if active_map:
                        if not full_path.name in layers_set:
                            active_map.addDataFromPath(str(full_path))
                        else:
                            arcpy.AddMessage('Le fichier est déjà dans la Carte')
                    else:
                        arcpy.AddWarning("Impossible d'ajouter à la carte car aucune carte n'est active.")

                except Exception as e:
                    arcpy.AddWarning(f"Impossible d'ouvrir/ajouter '{full_path}'. Erreur : {e}")
            else:
                arcpy.AddWarning(f"Fichier non trouvé : {full_path}")

except Exception as e:
    arcpy.AddError(f"Erreur lors de la lecture des entités sélectionnées : {e}")

# Calcul des limites de montaison (en utilisant un bundle 2 ou 3 variables)
# Script Python à utiliser en ligne de commande (voir exemples plus bas)
# Prérequis : geopandas, numpy, pandas, joblib (et le bundle .joblib sauvegardé après l’entraînement 2 ou 3 variables).
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Appliquer le modèle tacon (2 ou 3 variables : Surf_Drai [km²], max_slope_5 [m/m], Distance [m]) sur des shapefiles.

Fonctionnalités :
- Charge un bundle entraîné : {
    "final_calib": CalibratedClassifierCV,
    "features": ["Surf_Drai","max_slope_5"] ou ["Surf_Drai","max_slope_5","Distance"],
    "medians_train": dict,
    "thresholds": dict (nested {"loro": {...}, "loso": {...}} ou flat),
    "constraints": {"use_infran": bool, "slope5_threshold": float}
  }
- Lit un shapefile (.shp) ou GeoPackage (.gpkg) ou un dossier
- Mappe FACC_COR -> Surf_Drai (conversion m² -> km²)
- Mappe P_max_5  -> max_slope_5
- Mappe PK       -> Distance **en mètres** (gère formats '12+345' -> 12345 m)
- Applique contraintes (Infran + pente > seuil)
- Produit proba, proba_contrainte, contrainte_absence, pred (0/1)
- Écrit la sortie en .gpkg ou .shp

Exemples :
    python apply_tacon_model_gis.py \
      --input C:/rivieres/Riv1.shp \
      --output C:/rivieres/Riv1_pred.gpkg \
      --bundle C:/models/modele_tacon_3vars.joblib \
      --threshold f1 \
      --threshold_source loro \
      --layer Riv1 \
      --map_distance PK \
      --distance_unit m

    # Batch sur dossier :
    python apply_tacon_model_gis.py \
      --input C:/rivieres/ \
      --output C:/rivieres_pred/ \
      --bundle C:/models/modele_tacon_3vars.joblib \
      --threshold recall \
      --threshold_source loro \
      --map_distance PK \
      --distance_unit auto
"""

import os
import sys
import glob
import math
import argparse
import warnings
import unicodedata
from typing import Tuple, Dict, Optional, Union, List

import numpy as np
import pandas as pd
import geopandas as gpd
import joblib

# ----------------------------- A) Helpers normalisation & synonymes -----------------------------

def _norm(s: str) -> str:
    """Normalise un nom de champ : minuscule, sans accents, supprime espaces/underscores/traits/slashs."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    for ch in [" ", "_", "-", "/"]:
        s = s.replace(ch, "")
    return s

# Synonymes connus (clés normalisées) -> nom standard attendu
SYN_SURF = {
    "faccor": "Surf_Drai",      # surface de drainage en m²
    "facccorr": "Surf_Drai",
    "facc": "Surf_Drai",
    "surfdrai": "Surf_Drai",
    "surfdrain": "Surf_Drai",
}
SYN_SLOPE = {
    "pmax5": "max_slope_5",
    "pentemax5": "max_slope_5",
    "pente5": "max_slope_5",
    "maxslope5": "max_slope_5",
    "m_slope_5": "max_slope_5"
}
SYN_INFRAN = {
    "infran": "Infran",
    "infranavere": "Infran_Avere",
    "infranpoten": "Infran_Poten",
}
# ➕ Synonymes pour Distance (depuis embouchure trib.)
SYN_DISTANCE = {
    "distance": "Distance",
    "dist": "Distance",
    "distm": "Distance",
    "distance_m": "Distance",
    "distemb": "Distance",
    "distanceembouchure": "Distance",
    "distkm": "Distance",
    "distancekm": "Distance",
    "distembkm": "Distance",
    "distembm": "Distance",
}

def _find_src_for(target: str, synonyms: dict, columns: List[str]) -> Optional[str]:
    """Cherche un nom de colonne source compatible avec la cible (normalisation & synonymes)."""
    col_index = {_norm(c): c for c in columns}
    # Essai direct : une colonne a déjà le nom cible ?
    if _norm(target) in col_index:
        return col_index[_norm(target)]
    # Essai via synonymes connus
    for k_norm, std_name in synonyms.items():
        if std_name == target and k_norm in col_index:
            return col_index[k_norm]
    return None

# ----------------------------- B) Contraintes utilitaires -----------------------------

def _infer_infran_series(df: pd.DataFrame) -> pd.Series:
    """Retourne 1 si obstacle infranchissable détecté (Infran|Avere/Poten), sinon 0."""
    if "Infran" in df.columns:
        return df["Infran"].fillna(0).clip(0, 1).astype(int)
    has_avere = "Infran_Avere" in df.columns
    has_poten = "Infran_Poten" in df.columns
    if has_avere or has_poten:
        a = df["Infran_Avere"].fillna(0) if has_avere else 0
        p = df["Infran_Poten"].fillna(0) if has_poten else 0
        return (a + p).clip(0, 1).astype(int)
    return pd.Series(0, index=df.index, dtype=int)

def hard_constraints_mask(df: pd.DataFrame, slope5_threshold: float = 0.5, use_infran: bool = True) -> pd.Series:
    """True pour les lignes à forcer en absence (prob=0) via Infran et pente > seuil."""
    m = pd.Series(False, index=df.index)
    if use_infran:
        m = m | (_infer_infran_series(df) == 1)
    if ("max_slope_5" in df.columns) and (slope5_threshold is not None):
        m = m | (pd.to_numeric(df["max_slope_5"], errors="coerce") > float(slope5_threshold))
    return m

def apply_hard_constraints(df: pd.DataFrame, proba: np.ndarray,
                           slope5_threshold: float = 0.5, use_infran: bool = True) -> Tuple[np.ndarray, pd.Series]:
    """Met proba=0 si contrainte dure; retourne (proba_contrainte, mask 0/1)."""
    proba = np.asarray(proba, dtype=float).copy()
    mask = hard_constraints_mask(df, slope5_threshold=slope5_threshold, use_infran=use_infran)
    proba[mask.values] = 0.0
    return proba, mask.astype(int)

# ----------------------------- C) Lecture seuils (LOSO/LORO) -----------------------------

def _resolve_threshold(thresholds_obj, source: str, key: str) -> float:
    """
    Récupère un seuil en gérant :
      - bundle récent : thresholds = {"loso": {...}, "loro": {...}}
      - bundle ancien : thresholds = {"recall": x, "f1": y, "youden": z}
    Fallback : recall -> f1 -> youden -> 0.5
    """
    key = key.lower()
    source = source.lower()
    thr = None

    if isinstance(thresholds_obj, dict):
        # Cas 1 : bundle récent (nested dict)
        if source in thresholds_obj and isinstance(thresholds_obj[source], dict):
            thr = thresholds_obj[source].get(key, None)
        # Cas 2 : bundle ancien (flat dict)
        if thr is None and key in thresholds_obj:
            thr = thresholds_obj.get(key, None)
        # Fallbacks prudents
        if thr is None:
            for k in ["recall", "f1", "youden"]:
                v = (thresholds_obj[source].get(k)
                     if (source in thresholds_obj and isinstance(thresholds_obj[source], dict))
                     else thresholds_obj.get(k))
                if v is not None:
                    thr = v
                    break
    # Fallback final si rien
    if thr is None or (isinstance(thr, float) and math.isnan(thr)):
        thr = 0.5
    return float(thr)

# ----------------------------- D) I/O helpers -----------------------------

def read_any_vector(path: str, layer: Optional[str] = None) -> gpd.GeoDataFrame:
    """Lit un shapefile (.shp) ou GeoPackage (.gpkg)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".gpkg"):
        return gpd.read_file(path, layer=layer)
    else:
        return gpd.read_file(path)

def write_any_vector(gdf: gpd.GeoDataFrame, path: str, layer: Optional[str] = None):
    """Écrit un shapefile (.shp) ou GeoPackage (.gpkg)."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if path.lower().endswith(".gpkg"):
        gdf.to_file(path, layer=layer or "predictions", driver="GPKG")
    else:
        # Attention aux limites des shapefiles (noms de champs/encodage)
        gdf.to_file(path)

def list_vector_files(folder: str) -> list:
    """Retourne la liste des .shp et .gpkg dans un dossier (non récursif)."""
    shp = glob.glob(os.path.join(folder, "*.shp"))
    gpkg = glob.glob(os.path.join(folder, "*.gpkg"))
    return sorted(shp + gpkg)

# ----------------------------- E) Noyau prédiction GeoDataFrame -----------------------------

def predict_on_gdf(
    gdf: gpd.GeoDataFrame,
    bundle: Dict,
    colmap: Optional[Dict[str, str]] = None,
    convert_m2_to_km2: bool = True,
    use_threshold: Union[str, float] = "recall",
    threshold_source: str = "loro",  # par défaut LORO (recommandé inter-rivières)
    distance_unit: str = "m",        # 'm' (défaut), 'km', 'auto'
) -> Tuple[gpd.GeoDataFrame, float]:
    """
    Applique le modèle sur un GeoDataFrame.

    Paramètres
    ----------
    gdf : GeoDataFrame d'entrée (doit contenir FACC_COR (m²) et P_max_5; et Distance/PK si le bundle l'exige)
    bundle : dict chargé via joblib (final_calib, features, medians_train, thresholds, constraints)
    colmap : mapping explicite des colonnes source -> cibles du modèle
             ex: {"FACC_COR":"Surf_Drai","P_max_5":"max_slope_5","PK":"Distance","Infran":"Infran"}
    convert_m2_to_km2 : si True, convertit Surf_Drai m² -> km²
    use_threshold : "recall" | "f1" | "youden" | float
    threshold_source : "loro" | "loso" | "auto"
    distance_unit : 'm' (défaut) = garder en mètres ; 'km' = convertir km->m ; 'auto' = heuristique km->m

    Retour
    ------
    (gdf_out, thr_utilise)
    """
    final_calib     = bundle["final_calib"]
    features        = list(bundle["features"])        # ["Surf_Drai","max_slope_5"] ou ["Surf_Drai","max_slope_5","Distance"]
    medians_train   = bundle["medians_train"]
    thresholds      = bundle.get("thresholds", {})
    constraints_cfg = bundle.get("constraints", {"use_infran": True, "slope5_threshold": 0.5})

    gdf_proc = gdf.copy()

    # ---- helper local pour parser des PK '12+345' -> mètres ----
    def _parse_pk_to_m(val):
        """
        Convertit des formats PK (ex. '12+345', '12+345,6') en mètres.
        - '12+345'     -> 12345.0
        - '12+345,6'   -> 12345.6
        - nombre brut  -> renvoyé tel quel (sera interprété ensuite comme m ou km selon 'distance_unit')
        """
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, np.number)):
            return float(val)
        s = str(val).strip().replace(",", ".")
        if "+" in s:
            try:
                km_str, m_str = s.split("+", 1)
                km = float(km_str)
                m  = float(m_str)
                return km * 1000.0 + m
            except Exception:
                return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan

    # 1) Mapping colonnes source -> noms du modèle
    if colmap is None:
        colmap = {"FACC_COR": "Surf_Drai", "P_max_5": "max_slope_5"}  # + "Infran" si dispo
        if "Distance" in features and "distance" in gdf_proc.columns:
            colmap["distance"] = "Distance"

    # Appliquer mapping explicite si présent (source -> cible)
    for src_col, tgt_col in colmap.items():
        if src_col in gdf_proc.columns:
            gdf_proc[tgt_col] = gdf_proc[src_col]

    # Fallback tolérant si les cibles n'existent pas encore
    if "Surf_Drai" not in gdf_proc.columns:
        src = _find_src_for("Surf_Drai", SYN_SURF, list(gdf_proc.columns)) \
              or _find_src_for("FACC_COR", SYN_SURF, list(gdf_proc.columns))
        if src:
            gdf_proc["Surf_Drai"] = gdf_proc[src]

    if "max_slope_5" not in gdf_proc.columns:
        src = _find_src_for("max_slope_5", SYN_SLOPE, list(gdf_proc.columns)) \
              or _find_src_for("P_max_5", SYN_SLOPE, list(gdf_proc.columns))
        if src:
            gdf_proc["max_slope_5"] = gdf_proc[src]

    if "Distance" in features and "Distance" not in gdf_proc.columns:
        src = None
        if colmap:
            inv = {v: k for k, v in colmap.items()}  # support cible->source
            src = inv.get("Distance", None)
        if src is None:
            src = _find_src_for("Distance", SYN_DISTANCE, list(gdf_proc.columns))
        if src is not None and src in gdf_proc.columns:
            gdf_proc["Distance"] = gdf_proc[src]

    # 2) Conversion unités
    # Surf_Drai : m² -> km²
    if "Surf_Drai" not in gdf_proc.columns and "FACC_COR" in gdf_proc.columns:
        gdf_proc["Surf_Drai"] = gdf_proc["FACC_COR"]

    if "Surf_Drai" in gdf_proc.columns:
        gdf_proc["Surf_Drai"] = pd.to_numeric(gdf_proc["Surf_Drai"], errors="coerce")
        if convert_m2_to_km2:
            gdf_proc["Surf_Drai"] = gdf_proc["Surf_Drai"] / 1e6

    # Distance/PK -> mètres
    if "Distance" in gdf_proc.columns:
        col = gdf_proc["Distance"]
        if col.dtype == object:
            s = col.apply(_parse_pk_to_m)  # '12+345' -> 12345 m
        else:
            s = pd.to_numeric(col, errors="coerce")

        du = (distance_unit or "m").lower()
        if du not in ("m", "km", "auto"):
            warnings.warn(f"[WARN] distance_unit inconnu '{distance_unit}', utilisation de 'm'.")
            du = "m"

        if du == "km":
            s = s * 1000.0
        elif du == "auto":
            # Heuristique simple : si médiane <= 1000 ET max <= 2000 -> probable km -> convertir en mètres
            try:
                med_d = float(np.nanmedian(s.values))
                max_d = float(np.nanmax(s.values))
            except Exception:
                med_d, max_d = np.nan, np.nan
            if (not np.isnan(med_d) and med_d <= 1000.0) and (not np.isnan(max_d) and max_d <= 2000.0):
                s = s * 1000.0
                if max_d > 0:
                    warnings.warn("[INFO] distance_unit=auto : valeurs interprétées comme kilomètres -> converties en mètres.")
        # sinon 'm' => ne rien faire

        gdf_proc["Distance"] = s

    # 3) Vérifier colonnes attendues
    missing = [c for c in features if c not in gdf_proc.columns]
    if missing:
        sample_cols = list(gdf_proc.columns)[:15]
        print("[DIAG] Colonnes disponibles (échantillon) :", sample_cols)
        print("[DIAG] Colonnes normalisées (échantillon) :", [_norm(c) for c in sample_cols])
        raise ValueError(
            f"Colonnes manquantes après mapping/conversion : {missing} ; attendu={features}. "
            f"Vérifie les noms des champs ou utilise --map_FACC / --map_slope / --map_distance."
        )

    # 4) Imputation avec médianes du train
    X_new = gdf_proc[features].copy()
    for c in features:
        if X_new[c].isna().any():
            med = medians_train.get(c, np.nanmedian(X_new[c].values))
            X_new[c] = X_new[c].fillna(med)

    # Sanity checks
    if "Surf_Drai" in X_new.columns and (X_new["Surf_Drai"] < 0).any():
        warnings.warn("Valeurs négatives détectées pour Surf_Drai (km²).")
    if "max_slope_5" in X_new.columns and (X_new["max_slope_5"] < 0).any():
        warnings.warn("Valeurs négatives détectées pour max_slope_5.")
    if "Distance" in X_new.columns and (X_new["Distance"] < 0).any():
        warnings.warn("Valeurs négatives détectées pour Distance (m).")

    # 5) Probabilités calibrées
    proba = final_calib.predict_proba(X_new.values)[:, 1]

    # 6) Contraintes écologiques (Infran + pente)
    proba_constr, mask_constr = apply_hard_constraints(
        gdf_proc, proba,
        slope5_threshold=constraints_cfg.get("slope5_threshold", 0.5),
        use_infran=constraints_cfg.get("use_infran", True)
    )

    # 7) Seuil (LORO par défaut, rétro-compatibilité)
    if isinstance(use_threshold, (int, float)):
        thr = float(use_threshold)
        src_used = "custom_numeric"
    else:
        if threshold_source.lower() == "auto":
            try_sources = ["loro", "loso"]
        else:
            try_sources = [threshold_source.lower()]
        thr = None
        src_used = None
        for src in try_sources:
            thr_candidate = _resolve_threshold(thresholds, src, str(use_threshold))
            if thr_candidate is not None:
                thr = float(thr_candidate)
                src_used = src
                break
        if thr is None:
            thr = 0.5
            src_used = "fallback_0.5"

    pred = (proba_constr >= thr).astype(int)

    # 8) Sortie
    gdf_out = gdf.copy()
    gdf_out["Surf_Drai_km2"] = gdf_proc["Surf_Drai"]
    gdf_out["max_slope_5"]   = gdf_proc["max_slope_5"]
    if "Distance" in features:
        gdf_out["Distance_m"] = gdf_proc["Distance"]  # ✅ mètres
    gdf_out["proba"] = proba
    gdf_out["proba_contrainte"] = proba_constr
    gdf_out["contrainte_absence"] = mask_constr
    gdf_out["pred"] = pred
    gdf_out.attrs["seuil_utilise"] = thr
    gdf_out.attrs["source_seuil"] = src_used

    return gdf_out, thr

# ----------------------------- F) Main CLI -----------------------------

def _is_vector_file(path: str) -> bool:
    p = path.lower()
    return p.endswith(".shp") or p.endswith(".gpkg")

def _ensure_output_path_for_single_file(input_path: str, output_arg: str) -> str:
    """
    Si output_arg est un dossier ou un chemin sans extension, compose un .gpkg
    du type <basename>_pred.gpkg dans ce dossier. Sinon, renvoie output_arg tel quel.
    """
    if os.path.isdir(output_arg):
        base = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(output_arg, f"{base}_pred.gpkg")
    if not _is_vector_file(output_arg):
        out_dir = os.path.dirname(output_arg)
        base = os.path.splitext(os.path.basename(input_path))[0]
        if not out_dir:
            return os.path.abspath(f"{base}_pred.gpkg")
        else:
            os.makedirs(out_dir, exist_ok=True)
            return os.path.join(out_dir, f"{base}_pred.gpkg")
    return output_arg

def main():
    parser = argparse.ArgumentParser(description="Appliquer le modèle tacon (2 ou 3 variables) sur des shapefiles.")
    parser.add_argument("--input", required=True,
                        help="Chemin d'un shapefile/GeoPackage, ou d'un dossier contenant des .shp/.gpkg")
    parser.add_argument("--output", required=True,
                        help="Fichier de sortie (si input=fichier) ou dossier de sortie (si input=dossier). Recommandé : .gpkg")
    parser.add_argument("--bundle", required=True, help="Chemin du bundle .joblib (modèle + artefacts).")
    parser.add_argument("--threshold", default="recall",
                        help="Seuil : 'recall' | 'f1' | 'youden' | nombre (ex: 0.4).")
    parser.add_argument("--threshold_source", default="loro",
                        help="Source des seuils dans le bundle : 'loro' (défaut), 'loso' ou 'auto' (tente loro puis loso).")
    parser.add_argument("--layer", default=None, help="Nom de couche à lire/écrire pour GeoPackage.")
    parser.add_argument("--no_convert_km2", action="store_true",
                        help="Ne PAS convertir FACC_COR (m²) -> Surf_Drai (km²) (déconseillé).")
    parser.add_argument("--map_FACC", default="FACC_COR",
                        help="Nom du champ surface drainée (m²) dans le shapefile (ex.: FACC_COR).")
    parser.add_argument("--map_slope", default="P_max_5",
                        help="Nom du champ pente 5 m dans le shapefile (ex.: P_max_5).")
    parser.add_argument("--map_distance", default="distance",
                        help="Nom du champ Distance (ex.: PK). Interprété selon --distance_unit.")
    parser.add_argument("--map_infran", default=None,
                        help="Nom du champ Infran (optionnel). Sinon utiliser Infran_Avere/Poten si présents.")
    parser.add_argument("--distance_unit", default="m", choices=["m", "km", "auto"],
                        help="Unités de Distance/PK : 'm' (défaut), 'km' (converti en mètres), 'auto' (détecte km->m).")
    parser.add_argument("--verbose", action="store_true", help="Affiche des logs détaillés (recommandé pour debug).")
    args = parser.parse_args()

    bundle = joblib.load(args.bundle)  # ne pas écraser bundle['features'] : 2 ou 3 vars selon le modèle

    # Mapping colonnes (priorité aux arguments explicites)
    colmap = {
        args.map_FACC: "Surf_Drai",
        args.map_slope: "max_slope_5"
    }
    if args.map_infran:
        colmap[args.map_infran] = "Infran"
    if args.map_distance:
        colmap[args.map_distance] = "Distance"  # si le bundle n'utilise pas Distance, ce mapping sera ignoré

    convert = not args.no_convert_km2

    # CAS DOSSIER
    if os.path.isdir(args.input):
        in_files = list_vector_files(args.input)
        if not in_files:
            print(f"Aucun .shp/.gpkg trouvé dans : {args.input}")
            sys.exit(1)

        os.makedirs(args.output, exist_ok=True)
        ok, ko = 0, 0
        for f in in_files:
            try:
                gdf = read_any_vector(f, layer=args.layer)
                gdf_out, thr = predict_on_gdf(
                    gdf, bundle, colmap=colmap, convert_m2_to_km2=convert,
                    use_threshold=args.threshold, threshold_source=args.threshold_source,
                    distance_unit=args.distance_unit
                )
                base = os.path.splitext(os.path.basename(f))[0]
                out_path = os.path.join(args.output, f"{base}_pred.gpkg")
                write_any_vector(gdf_out, out_path, layer=args.layer or base)
                if args.verbose:
                    print(f"[OK] {os.path.abspath(f)} -> {os.path.abspath(out_path)} | "
                          f"n={len(gdf_out)} | thr={thr:.3f} | src={gdf_out.attrs.get('source_seuil')} | "
                          f"feats={bundle.get('features')} | dist_unit={args.distance_unit}")
                ok += 1
            except Exception as e:
                print(f"[ERREUR] {f} : {e}")
                ko += 1

        print(f"\nTerminé. Succès: {ok} | Échecs: {ko} | Sortie: {os.path.abspath(args.output)}")
        return

    # CAS FICHIER UNIQUE
    gdf = read_any_vector(args.input, layer=args.layer)
    out_path = _ensure_output_path_for_single_file(args.input, args.output)

    gdf_out, thr = predict_on_gdf(
        gdf, bundle, colmap=colmap, convert_m2_to_km2=convert,
        use_threshold=args.threshold, threshold_source=args.threshold_source,
        distance_unit=args.distance_unit
    )

    write_any_vector(gdf_out, out_path, layer=args.layer)

    if args.verbose:
        print(f"[OK] Input : {os.path.abspath(args.input)}")
        print(f"[OK] Output: {os.path.abspath(out_path)} (n={len(gdf_out)}, thr={thr:.3f}, "
              f"src={gdf_out.attrs.get('source_seuil')}, feats={bundle.get('features')}, dist_unit={args.distance_unit})")

    if os.path.exists(out_path):
        print(f"✅ Écrit : {out_path} (seuil utilisé = {thr:.3f}, source = {gdf_out.attrs.get('source_seuil')})")
    else:
        print("⚠️ Avertissement : impossible de confirmer la création du fichier de sortie.")

if __name__ == "__main__":
    main()

"""
Script de modelado predictivo espacial para índices de vegetación y agua.

Este módulo incorpora dos escenarios principales basados en literatura
científica resumida por el usuario:

- Escenario de degradación moderada: precipitación −20%, temperatura
  +2 °C, radiación +2%, PET +20%, ET −10% y escorrentía +20% (aumentos y
  reducciones coherentes con los rangos IPCC/regionales citados).
- Escenario de conservación moderada: precipitación +10%, temperatura
  −1 °C (mitiga el calentamiento local), radiación −2%, PET −10%, ET
  +10% y escorrentía −10%, representando restauración hídrica y
  microclima más fresco.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# =========================================================
#                    CONFIGURACIÓN
# =========================================================

# Carpeta raíz del proyecto
BASE_DIR = r"C:\Users\estudianteap2\Downloads\Imagenes satelitales"

# Carpeta de ráster
RASTERS_DIR = os.path.join(BASE_DIR, "Raters")  # Cambia a "Rasters" si es tu caso

# Carpetas de vegetación y río
VEG_DIR = os.path.join(RASTERS_DIR, "VEGETACION")
RIO_DIR = os.path.join(RASTERS_DIR, "RIO")

# Clima
CLIMA_CSV = os.path.join(BASE_DIR, "Datos", "Clima_ERA5Land_2000_2025.csv")

# Variables climáticas que vamos a usar
CLIMATE_VARS = ["P_mm", "Tmean_C", "PET_mm", "ET_mm", "Rad_MJm2", "Runoff_mm"]

# Índices por zona
INDICES_VEG = ["NDVI", "NDMI", "NBR"]
INDICES_RIO = ["MNDWI"]

# Número máximo de píxeles de entrenamiento por ráster
MAX_SAMPLES_PER_RASTER = 5000

# Carpeta donde guardamos predicciones
OUT_PRED_DIR = os.path.join(RASTERS_DIR, "PREDICCIONES")


@dataclass(frozen=True)
class ScenarioAdjustment:
    """Ajustes climáticos e índices para un escenario."""

    precip_factor: float
    temp_offset: float
    rad_factor: float
    pet_factor: float
    et_factor: float
    runoff_factor: float
    veg_index_factor: Dict[str, float]
    rio_index_factor: Dict[str, float]


SCENARIOS: Dict[str, ScenarioAdjustment] = {
    # Valores moderados solicitados (derivados de la síntesis de literatura):
    # Degradación: P −20%, T +2 °C, Rad +2%, PET +20%, ET −10%, Runoff +20%.
    "degradacion": ScenarioAdjustment(
        precip_factor=0.8,
        temp_offset=2.0,
        rad_factor=1.02,
        pet_factor=1.20,
        et_factor=0.90,
        runoff_factor=1.20,
        veg_index_factor={"NDVI": 0.85, "NDMI": 0.6, "NBR": 0.8},
        rio_index_factor={"MNDWI": 0.85},
    ),
    # Conservación: P +10%, T −1 °C, Rad −2%, PET −10%, ET +10%, Runoff −10%.
    "conservacion": ScenarioAdjustment(
        precip_factor=1.10,
        temp_offset=-1.0,
        rad_factor=0.98,
        pet_factor=0.90,
        et_factor=1.10,
        runoff_factor=0.90,
        veg_index_factor={"NDVI": 1.15, "NDMI": 1.20, "NBR": 1.05},
        rio_index_factor={"MNDWI": 1.15},
    ),
}


# =========================================================
#                 FUNCIONES AUXILIARES
# =========================================================

def cargar_clima() -> pd.DataFrame:
    """Carga el CSV de clima y devuelve un DataFrame."""
    clima = pd.read_csv(CLIMA_CSV)
    # Normalizamos nombres en minúsculas por si acaso
    clima.columns = [c.strip() for c in clima.columns]
    return clima


def listar_rasters_indice_zona(indice: str, zona: str) -> Iterable[str]:
    """
    Lista todos los ráster disponibles para un índice y zona.
    Ejemplo:
      indice="NDVI", zona="VEGETACION"
      ruta ~ VEGETACION/NDVI/**/NDVI_VEG_YYYY_MM.tif
    """
    if zona.upper() == "VEGETACION":
        base = os.path.join(VEG_DIR, indice)
        zona_code = "VEG"
    else:
        base = os.path.join(RIO_DIR, indice)
        zona_code = "RIO"

    pattern = os.path.join(base, "**", f"{indice}_{zona_code}_*.tif")
    files = glob.glob(pattern, recursive=True)
    files = sorted(files)
    return files


def parse_year_month_from_name(path: str) -> Tuple[int, int]:
    """
    Extrae año y mes del nombre del archivo.
    Formato esperado: INDICE_VEG_YYYY_MM.tif
    Ej: NDVI_VEG_2019_01.tif
    """
    name = os.path.basename(path)
    name_noext = os.path.splitext(name)[0]
    parts = name_noext.split("_")
    # [..., YYYY, MM]
    year = int(parts[-2])
    month = int(parts[-1])
    return year, month


def raster_to_samples(
    path_raster: str,
    clima_df: pd.DataFrame,
    indice: str,
    max_samples: int = MAX_SAMPLES_PER_RASTER,
):
    """
    Convierte un ráster (una fecha) en muestras (X, y) para entrenamiento.
    X: [x, y, clima..., year, month]
    y: valor de índice en el píxel
    """
    year, month = parse_year_month_from_name(path_raster)

    # Buscar clima de ese mes
    row = clima_df[(clima_df["year"] == year) & (clima_df["month"] == month)]
    if row.empty:
        print(f"[AVISO] No hay clima para {year}-{month:02d}, saltando {path_raster}")
        return None, None

    row = row.iloc[0]

    # Abrir ráster
    with rasterio.open(path_raster) as src:
        data = src.read(1).astype("float32")
        nodata = src.nodata
        transform: Affine = src.transform
        height, width = data.shape

    # Máscara de píxeles válidos
    arr = data.copy()
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    valid_mask = np.isfinite(arr)
    valid_indices = np.where(valid_mask.ravel())[0]

    if len(valid_indices) == 0:
        print(f"[AVISO] No hay píxeles válidos en {path_raster}")
        return None, None

    # Elegir un subconjunto aleatorio de píxeles
    n_samples = min(max_samples, len(valid_indices))
    chosen = np.random.choice(valid_indices, size=n_samples, replace=False)

    ys = arr.ravel()[chosen]

    # Obtener filas y columnas de esos índices
    rows = chosen // width
    cols = chosen % width

    # Convertir a coordenadas x, y
    xs = []
    ys_coord = []
    for r, c in zip(rows, cols):
        x, y = transform * (c + 0.5, r + 0.5)  # centro del píxel
        xs.append(x)
        ys_coord.append(y)

    xs = np.array(xs)
    ys_coord = np.array(ys_coord)

    # Features climáticos (constantes para todos los píxeles de este mes)
    clima_features = np.array([[row[v] for v in CLIMATE_VARS]] * n_samples)

    # Años y meses (también constantes por ráster)
    years = np.full((n_samples, 1), year, dtype="int32")
    months = np.full((n_samples, 1), month, dtype="int32")

    # Construimos la matriz X
    X = np.column_stack([
        xs,
        ys_coord,
        clima_features,
        years,
        months,
    ])

    y = ys  # valores del índice

    return X, y


def construir_dataset(indice: str, zona: str, clima_df: pd.DataFrame):
    """
    Recorre todos los ráster del índice y zona y arma un dataset de entrenamiento.
    """
    print(f"\n=== Construyendo dataset para {indice} - {zona} ===")
    files = list(listar_rasters_indice_zona(indice, zona))
    print(f"Ráster encontrados: {len(files)}")

    X_list = []
    y_list = []

    for path in files:
        print(f"  Procesando {path} ...")
        Xi, yi = raster_to_samples(path, clima_df, indice)
        if Xi is None:
            continue
        X_list.append(Xi)
        y_list.append(yi)

    if not X_list:
        raise RuntimeError(f"No se pudo construir dataset para {indice}-{zona}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print(f"Dataset {indice}-{zona}: {X.shape[0]} muestras, {X.shape[1]} features")
    return X, y


def entrenar_modelo_indice(indice: str, zona: str, clima_df: pd.DataFrame):
    """
    Entrena un RandomForestRegressor para un índice y zona.
    """
    X, y = construir_dataset(indice, zona, clima_df)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(Xtrain, ytrain)

    r2 = model.score(Xtest, ytest)
    print(f"Modelo {indice}-{zona}: R² = {r2:.3f}")

    return model


def entrenar_modelos(clima_df: pd.DataFrame):
    """
    Entrena todos los modelos para vegetación y río.
    """
    modelos_veg = {}
    modelos_rio = {}

    for ind in INDICES_VEG:
        modelos_veg[ind] = entrenar_modelo_indice(ind, "VEGETACION", clima_df)

    for ind in INDICES_RIO:
        modelos_rio[ind] = entrenar_modelo_indice(ind, "RIO", clima_df)

    return modelos_veg, modelos_rio


# =========================================================
#        PREDICCIÓN ESPACIAL (GENERACIÓN DE RÁSTER)
# =========================================================

def obtener_clima(clima_df: pd.DataFrame, year: int, month: int) -> pd.Series:
    row = clima_df[(clima_df["year"] == year) & (clima_df["month"] == month)]
    if not row.empty:
        return row.iloc[0]

    # Si no hay clima futuro, usar la referencia más reciente del mismo mes
    mensual = clima_df[clima_df["month"] == month]
    if mensual.empty:
        raise ValueError(
            f"No hay clima disponible para el mes {month:02d} en el dataset"
        )

    row_fallback = mensual.loc[mensual["year"].idxmax()]
    print(
        f"[AVISO] No hay clima para {year}-{month:02d}. "
        f"Usando clima de referencia {int(row_fallback['year'])}-{int(row_fallback['month']):02d} "
        "y aplicando ajustes de escenario."
    )
    return row_fallback


def elegir_raster_plantilla(indice: str, zona: str) -> str:
    """
    Para predecir, necesitamos la grilla (shape, transform, CRS).
    Usamos cualquier ráster existente del índice y zona como plantilla.
    """
    files = list(listar_rasters_indice_zona(indice, zona))
    if not files:
        raise RuntimeError(f"No hay ráster para usar como plantilla en {indice}-{zona}")
    # elegimos el último (por ejemplo)
    return files[-1]


def aplicar_ajustes_climaticos(row: pd.Series, escenario: str) -> pd.Series:
    """Devuelve una copia de la fila climática ajustada según el escenario."""
    if escenario not in SCENARIOS:
        raise ValueError(f"Escenario no reconocido: {escenario}")

    cfg = SCENARIOS[escenario]
    row_adj = row.copy()
    row_adj["P_mm"] = row["P_mm"] * cfg.precip_factor
    row_adj["Tmean_C"] = row["Tmean_C"] + cfg.temp_offset
    row_adj["Rad_MJm2"] = row["Rad_MJm2"] * cfg.rad_factor
    row_adj["PET_mm"] = row["PET_mm"] * cfg.pet_factor
    row_adj["ET_mm"] = row["ET_mm"] * cfg.et_factor
    row_adj["Runoff_mm"] = row["Runoff_mm"] * cfg.runoff_factor
    return row_adj


def _coordenadas_pixeles(transform: Affine, height: int, width: int):
    """Genera vectores de coordenadas x, y para toda la grilla."""
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    cols_flat = cols.ravel()
    rows_flat = rows.ravel()

    xs = np.zeros_like(cols_flat, dtype="float64")
    ys = np.zeros_like(rows_flat, dtype="float64")
    for i, (c, r) in enumerate(zip(cols_flat, rows_flat)):
        x, y = transform * (c + 0.5, r + 0.5)
        xs[i] = x
        ys[i] = y
    return xs, ys


def generar_raster_predicho(
    indice: str,
    zona: str,
    model: RandomForestRegressor,
    clima_df: pd.DataFrame,
    year: int,
    month: int,
    escenario: str,
):
    """
    Genera un ráster completo para un índice/zona/año/mes bajo un escenario.
    - "degradacion": condiciones secas y cálidas
    - "conservacion": condiciones húmedas y frescas
    """
    clima_row = obtener_clima(clima_df, year, month)
    clima_row = aplicar_ajustes_climaticos(clima_row, escenario)
    path_template = elegir_raster_plantilla(indice, zona)

    with rasterio.open(path_template) as src:
        profile = src.profile
        transform = src.transform
        height, width = src.height, src.width

    xs, ys_coord = _coordenadas_pixeles(transform, height, width)

    # Features climáticos (iguales para todo el raster, pero repetidos)
    clima_features = np.array([[clima_row[v] for v in CLIMATE_VARS]] * len(xs))

    years = np.full((len(xs), 1), year, dtype="int32")
    months = np.full((len(xs), 1), month, dtype="int32")

    X_pred = np.column_stack([
        xs,
        ys_coord,
        clima_features,
        years,
        months,
    ])

    # Predicción
    y_pred = model.predict(X_pred)

    cfg = SCENARIOS[escenario]
    if zona.upper() == "VEGETACION":
        factor = cfg.veg_index_factor.get(indice, 1.0)
    else:
        factor = cfg.rio_index_factor.get(indice, 1.0)

    y_pred = y_pred * factor

    # recortar a [-1, 1] por ser índices normalizados
    y_pred = np.clip(y_pred, -1.0, 1.0)

    # Convertir de vector a matriz
    arr_pred = y_pred.reshape((height, width)).astype("float32")

    # Guardar ráster
    os.makedirs(OUT_PRED_DIR, exist_ok=True)

    zona_code = "VEG" if zona.upper() == "VEGETACION" else "RIO"
    esc_code = "DEG" if escenario == "degradacion" else "CON"
    out_name = f"{indice}_{zona_code}_{esc_code}_{year}_{month:02d}.tif"
    out_path = os.path.join(OUT_PRED_DIR, out_name)

    profile_out = profile.copy()
    profile_out.update(dtype="float32", count=1, compress="lzw")

    with rasterio.open(out_path, "w", **profile_out) as dst:
        dst.write(arr_pred, 1)

    print(f"Ráster generado: {out_path}")
    return out_path


def generar_todos_indices_para_mes(
    modelos_veg,
    modelos_rio,
    clima_df,
    year: int,
    month: int,
):
    """
    Genera ráster para todos los índices y para los 2 escenarios (degradación y conservación).
    """
    rutas = {"degradacion": {}, "conservacion": {}}

    # Vegetación
    for ind in INDICES_VEG:
        model = modelos_veg[ind]
        for esc in rutas.keys():
            rutas[esc][f"VEG_{ind}"] = generar_raster_predicho(
                indice=ind,
                zona="VEGETACION",
                model=model,
                clima_df=clima_df,
                year=year,
                month=month,
                escenario=esc,
            )

    # Río
    for ind in INDICES_RIO:
        model = modelos_rio[ind]
        for esc in rutas.keys():
            rutas[esc][f"RIO_{ind}"] = generar_raster_predicho(
                indice=ind,
                zona="RIO",
                model=model,
                clima_df=clima_df,
                year=year,
                month=month,
                escenario=esc,
            )

    return rutas


# =========================================================
#                     MAIN
# =========================================================

if __name__ == "__main__":
    print("=== MODELO PREDICTIVO ESPACIAL (ESCENARIOS DEGRADACIÓN/CONSERVACIÓN) ===")

    # 1. Año/mes a predecir
    target_year = int(
        input(
            "Año futuro a predecir (si no está en el CSV se usará el mes más reciente): "
        )
    )
    target_month = int(input("Mes futuro (1-12): "))

    # 2. Cargar clima
    clima_df = cargar_clima()

    # (opcional) filtrar clima al rango de años de tus ráster: 2019-2025
    # clima_df = clima_df[(clima_df["year"] >= 2019) & (clima_df["year"] <= 2025)]

    # 3. Entrenar modelos
    modelos_veg, modelos_rio = entrenar_modelos(clima_df)

    # 4. Generar rasters para el mes/escenarios
    rutas = generar_todos_indices_para_mes(
        modelos_veg, modelos_rio, clima_df, target_year, target_month
    )

    print("\n=== ESCENARIO DEGRADACIÓN ===")
    for k, v in rutas["degradacion"].items():
        print(k, "->", v)

    print("\n=== ESCENARIO CONSERVACIÓN ===")
    for k, v in rutas["conservacion"].items():
        print(k, "->", v)

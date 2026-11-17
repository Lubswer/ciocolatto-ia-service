from datetime import date, datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS


def _construir_serie_continua(
    historial: List[Dict[str, Any]],
    fecha_fin: date
) -> pd.DataFrame:
    """Convierte el historial disperso en una serie diaria continua rellenando huecos con 0."""
    if not historial:
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.DataFrame(historial)
    df["ds"] = pd.to_datetime(df["fecha"]).dt.date
    df = df.groupby("ds", as_index=False)["cantidad"].sum()

    fecha_inicio = df["ds"].min()
    if fecha_inicio > fecha_fin:
        fecha_fin = fecha_inicio

    rango_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="D").date
    mapa_ventas = dict(zip(df["ds"], df["cantidad"]))

    filas = [{"ds": pd.Timestamp(d), "y": float(mapa_ventas.get(d, 0))} for d in rango_completo]
    return pd.DataFrame(filas)


def _prediccion_promedio_conservador(
    serie: pd.DataFrame,
    fecha_inicio_pred: date,
    dias: int,
    margen_ratio: float
) -> List[Dict[str, Any]]:
    """Genera una proyección conservadora basada en el promedio reciente (hasta últimos 7 días)."""
    if serie.empty:
        promedio = 0.0
    else:
        ventana = serie.tail(7)
        promedio = float(ventana["y"].mean())

    esperado = max(0, int(round(promedio)))
    margen = max(1, int(round(esperado * margen_ratio))) if esperado > 0 else 0
    minimo = max(0, esperado - margen)
    maximo = esperado + margen

    return [
        {
            "fecha": (fecha_inicio_pred + timedelta(days=i)).isoformat(),
            "cantidad_esperada": esperado,
            "minimo": minimo,
            "maximo": maximo,
        }
        for i in range(dias)
    ]


def ejecutar_warmup() -> None:
    """Pre-compila numba/StatsForecast al arrancar el servidor para evitar cold start > 3s."""
    fechas = pd.date_range(end=pd.Timestamp.today().normalize(), periods=21, freq="D")
    df_dummy = pd.DataFrame({
        "unique_id": ["warmup"] * 21,
        "ds": fechas,
        "y": [5, 4, 6, 5, 8, 12, 14] * 3,
    })
    sf = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=1)
    sf.forecast(df=df_dummy, h=3, level=[80])


def generar_predicciones(
    productos: List[Dict[str, Any]],
    dias: int,
    fecha_corte_str: str | None = None
) -> List[Dict[str, Any]]:
    fecha_corte = (
        datetime.strptime(fecha_corte_str, "%Y-%m-%d").date()
        if fecha_corte_str
        else (date.today() - timedelta(days=1))
    )
    fecha_inicio_pred = fecha_corte + timedelta(days=1)

    resultados: List[Dict[str, Any]] = []
    frames_ets: List[pd.DataFrame] = []
    series_por_id: Dict[int, pd.DataFrame] = {}
    meta_por_id: Dict[int, Dict[str, Any]] = {}

    for prod in productos:
        pid = int(prod["producto_id"])
        es_temporada = bool(prod.get("producto_temporada", False))
        serie = _construir_serie_continua(prod.get("historial", []), fecha_corte)
        series_por_id[pid] = serie
        meta_por_id[pid] = prod

        # Regla 1: Producto de temporada -> estimación conservadora de corto plazo
        if es_temporada:
            preds = _prediccion_promedio_conservador(serie, fecha_inicio_pred, dias, 0.25)
            resultados.append({
                "producto_id": pid,
                "estrategia": "temporada",
                "total_esperado": sum(p["cantidad_esperada"] for p in preds),
                "predicciones": preds,
            })
            continue

        # Regla 2: Menos de 14 días de historia -> promedio simple
        if len(serie) < 14:
            preds = _prediccion_promedio_conservador(serie, fecha_inicio_pred, dias, 0.30)
            resultados.append({
                "producto_id": pid,
                "estrategia": "pocos_datos",
                "total_esperado": sum(p["cantidad_esperada"] for p in preds),
                "predicciones": preds,
            })
            continue

        # Regla 3: Producto regular con >= 14 días -> AutoETS estacional semanal
        df_prod = serie.copy()
        df_prod["unique_id"] = str(pid)
        frames_ets.append(df_prod[["unique_id", "ds", "y"]])

    if frames_ets:
        df_lote = pd.concat(frames_ets, ignore_index=True)
        sf = StatsForecast(
            models=[AutoETS(season_length=7)],
            freq="D",
            n_jobs=1,
        )
        df_pred = sf.forecast(df=df_lote, h=dias, level=[80]).reset_index()

        for uid_str, grupo in df_pred.groupby("unique_id"):
            pid = int(uid_str)
            preds: List[Dict[str, Any]] = []

            for _, fila in grupo.sort_values("ds").iterrows():
                val_raw = float(fila.get("AutoETS", 0) or 0)
                lo_raw = float(fila.get("AutoETS-lo-80", val_raw) or 0)
                hi_raw = float(fila.get("AutoETS-hi-80", val_raw) or 0)

                esperado = max(0, int(round(val_raw)))
                minimo = min(esperado, max(0, int(round(lo_raw))))
                maximo = max(esperado, int(round(hi_raw)))

                preds.append({
                    "fecha": pd.Timestamp(fila["ds"]).date().isoformat(),
                    "cantidad_esperada": esperado,
                    "minimo": minimo,
                    "maximo": maximo,
                })

            resultados.append({
                "producto_id": pid,
                "estrategia": "auto_ets",
                "total_esperado": sum(p["cantidad_esperada"] for p in preds),
                "predicciones": preds,
            })

    return resultados

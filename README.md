# Manufacturing Finance Random Forest 8+4

Aplicación bilingüe de Streamlit en modo oscuro para pronosticar septiembre-diciembre mediante Random Forest y clasificar las desviaciones como Favorable, En riesgo o Desfavorable.

## Funciones
- Carga manual de Excel (`Actuals`, `PY`, `PM Database`) o CSV normalizado.
- Actuals enero-agosto, pronóstico septiembre-diciembre.
- Comparación contra Prior Year 2025 y Current Forecast 8+4.
- Filtros BU/POD, planta y Cost Bucket.
- Modo ejecutivo y avanzado.
- Umbrales ajustables de porcentaje y materialidad.
- Intervalos obtenidos de la dispersión de árboles.
- Escenario what-if.
- Calidad de datos, métricas e importancia de variables.
- Descargas en Excel, CSV, JSON y Joblib.

## Seguridad de datos
El archivo financiero original no debe publicarse en GitHub. `.gitignore` excluye archivos Excel y CSV. La aplicación recibe la base mediante carga manual en Streamlit.

## Estructura
```text
app.py
requirements.txt
README.md
.gitignore
.streamlit/config.toml
```

## Ejecución local
1. Instala Python 3.11 o superior.
2. Crea y activa un entorno virtual.
3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecuta:
   ```bash
   streamlit run app.py
   ```
5. Carga `2026 Performance Model Consolidation 8+4.1.xlsx` en la interfaz.

## Publicación en GitHub y Streamlit Community Cloud
1. Crea un repositorio privado.
2. Sube únicamente los archivos del proyecto. No subas la base financiera.
3. En Streamlit Community Cloud, crea una app desde el repositorio.
4. Define `app.py` como archivo principal.
5. Abre la aplicación y carga el Excel manualmente.

## Estructura esperada del Excel
- `Actuals`: actuals 2026 y columnas mensuales.
- `PY`: Prior Year 2025 y columnas mensuales.
- `PM Database`: Current Forecast 8+4 y columnas mensuales.

La aplicación busca columnas equivalentes a Plant/Site, Cost Bucket/Database Cost Bucket, BU/POD, Cost Type y meses Jan-Dec o January-December.

## CSV normalizado
Columnas requeridas:
`Plant, CostBucket, BU, Year, Month, Value, Source`

Columnas recomendadas:
`Scenario, DataType, CostType`

## Nota metodológica
El objetivo numérico se modela con `RandomForestRegressor`. La clasificación ejecutiva se calcula después de predecir, usando dirección financiera, porcentaje de desviación y materialidad. Si no existen actuals observados para septiembre-diciembre, la aplicación usa una aproximación PY/Forecast y lo indica como modo proxy; para un backtesting temporal verdadero se requieren vintages históricos 8+4 junto con sus resultados observados.

## English summary
Dark-mode bilingual Streamlit app for Sep-Dec manufacturing-finance forecasting. Upload the confidential workbook manually, train the Random Forest model, compare results with PY and current 8+4, explore uncertainty and feature importance, and download analytical outputs.

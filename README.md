# Entrenamiento de calidad de vino con MLflow

`train.py` ejecuta un experimento reproducible de clasificación de la calidad
del vino tinto utilizando regresión logística, Scikit-learn y MLflow Tracking.
El objetivo (`quality`) se predice a partir de las características fisicoquímicas
del archivo CSV de vinos tintos de la UCI.

## Qué hace el script

El flujo de entrenamiento realiza los siguientes pasos:

1. Carga el dataset desde una ruta local o desde una URL. Por defecto utiliza
   el dataset `winequality-red.csv` de la UCI.
2. Elimina filas duplicadas y muestra el número de registros antes y después
   de la limpieza.
3. Separa las variables predictoras de la columna `quality`.
4. Divide los datos en entrenamiento (80 %) y prueba (20 %) usando partición
   estratificada.
5. Estandariza las variables con `StandardScaler` dentro de un `Pipeline`.
6. Entrena una `LogisticRegression` con los hiperparámetros indicados.
   Cuando se selecciona `liblinear`, el clasificador se envuelve en
   `OneVsRestClassifier` para permitir la clasificación multicategoría.
7. Calcula `accuracy` y F1 ponderado sobre el conjunto de prueba.
8. Genera la matriz de confusión en
   `outputs/matriz_confusion.png`.
9. Registra en MLflow los parámetros, métricas, matriz de confusión y pipeline
   entrenado.

## Requisitos

- Python 3.14 o posterior.
- Dependencias listadas en [requirements.txt](requirements.txt).
- Acceso a Internet si se utiliza la URL predeterminada del dataset.

### Instalación del entorno

Desde esta carpeta, crea y activa un entorno virtual e instala las
dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, activa el entorno con:

```powershell
.venv\Scripts\activate
```

## Uso

Ejecución con los valores predeterminados:

```bash
python train.py
```

Ejemplo con los parámetros usados para ejecutar `liblinear` en un problema
multicategoría:

```bash
python train.py \
  --c_param 1.0 \
  --max_iter 1000 \
  --solver liblinear
```

Ejemplo activando el balanceo de clases:

```bash
python train.py \
  --solver lbfgs \
  --class_weight balanced \
  --experiment_name Wine_Quality_Balanced
```

Para usar un archivo local, el CSV debe tener separador `;` y una columna
llamada `quality`:

```bash
python train.py \
  --data_path data/winequality-red.csv \
  --random_state 42 \
  --experiment_name Wine_Quality_Local
```

## Argumentos disponibles

| Argumento | Valor predeterminado | Descripción |
| --- | --- | --- |
| `--data_path` | URL del dataset UCI | Ruta local o URL del CSV. |
| `--c_param` | `1.0` | Inverso de la fuerza de regularización. |
| `--max_iter` | `1000` | Número máximo de iteraciones del optimizador. |
| `--solver` | `lbfgs` | Solver de LogisticRegression: `lbfgs`, `liblinear`, `saga` o `newton-cg`. |
| `--class_weight` | `none` | Usa `none` o `balanced` para ponderar las clases. |
| `--random_state` | `42` | Semilla para la partición y el entrenamiento. |
| `--experiment_name` | `Wine_Quality_Baseline_AlejandroFigueroa` | Nombre del experimento en MLflow. |

También se puede consultar esta lista directamente con:

```bash
python train.py --help
```

## Resultados y MLflow

Durante la ejecución se imprimen en consola el tamaño del dataset, la
configuración del modelo y las métricas calculadas. El archivo de imagen se
guarda localmente en:

```text
outputs/matriz_confusion.png
```

MLflow registra cada ejecución como una corrida del experimento indicado en
`--experiment_name`. Para consultar las corridas con la interfaz local de
MLflow, inicia el servidor usando el backend configurado para este proyecto.
Por ejemplo, si se está utilizando la base SQLite incluida:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Después abre `http://127.0.0.1:5000` en el navegador. Si el entorno usa otra
configuración de tracking, inicia la interfaz con el mismo backend o URI de
tracking utilizado por la ejecución del script.
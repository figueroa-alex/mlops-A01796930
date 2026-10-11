# =============================================================================
# SCRIPT DE EXPERIMENTACIÓN CON MLFLOW TRACKING
# Materia: TC5061 - MLOps
# Alumno: Alejandro Figueroa
# =============================================================================

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Módulo de MLflow
import mlflow
import mlflow.sklearn


# -----------------------------------------------------------------------------
# PASO 1: CARGA Y LIMPIEZA DE DATOS (REPLICANDO EL BASELINE)
# -----------------------------------------------------------------------------
def cargar_y_limpiar_datos(ruta_datos):
    """
    Carga el archivo CSV delimitado por punto y coma y elimina registros
    duplicados, garantizando consistencia con el notebook base (1,359 filas).
    """
    print(f"[DATOS] Cargando dataset desde: {ruta_datos}")
    df = pd.read_csv(ruta_datos, sep=";")
    
    filas_iniciales = len(df)
    df_limpio = df.drop_duplicates().copy()
    filas_removidas = filas_iniciales - len(df_limpio)
    
    print(f"[DATOS] Registros cargados: {filas_iniciales}")
    print(f"[DATOS] Duplicados removidos: {filas_removidas}")
    print(f"[DATOS] Registros finales para modelado: {len(df_limpio)}")
    
    return df_limpio


# -----------------------------------------------------------------------------
# PASO 2: PARTICIÓN DE DATOS ESTRATIFICADA
# -----------------------------------------------------------------------------
def partir_datos(df, semilla):
    """
    Divide los datos en 80% entrenamiento y 20% prueba de forma estratificada.
    """
    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=semilla,
        stratify=y
    )

    print(f"[DATOS] Muestras en entrenamiento: {len(X_train)} | Muestras en prueba: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# PASO 3: GENERACIÓN Y GUARDADO DE ARTEFACTO VISUAL
# -----------------------------------------------------------------------------
def generar_artefacto_matriz(y_real, y_pred, ruta_archivo="outputs/matriz_confusion.png"):
    """
    Crea un mapa de calor con la matriz de confusión y lo guarda como imagen PNG
    para registrarlo formalmente como artefacto en MLflow.
    """
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    
    matriz = confusion_matrix(y_real, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusión - Conjunto de Prueba")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()
    
    plt.savefig(ruta_archivo)
    plt.close()
    print(f"[ARTEFACTO] Gráfico guardado exitosamente en: {ruta_archivo}")
    return ruta_archivo


# -----------------------------------------------------------------------------
# PASO 4: ENTRENAMIENTO, EVALUACIÓN Y REGISTRO EN MLFLOW
# -----------------------------------------------------------------------------
def entrenar_y_registrar(ruta_datos, c_param, max_iter, solver, class_weight, semilla, nombre_experimento):
    """
    Ejecuta el ciclo de entrenamiento y registra parámetros, métricas,
    artefactos y el modelo dentro de MLflow Tracking.
    """
    # 1. Definir o recuperar el experimento
    mlflow.set_experiment(nombre_experimento)

    # 2. Iniciar el contexto de la corrida
    with mlflow.start_run():
        print(f"\n[MLFLOW] Iniciando corrida en experimento: '{nombre_experimento}'")

        # Cargar datos y realizar partición
        df = cargar_y_limpiar_datos(ruta_datos)
        X_train, X_test, y_train, y_test = partir_datos(df, semilla)

        # Scikit-Learn espera None (sin comillas) o el texto "balanced"
        peso_clases = None if class_weight == "none" else "balanced"

        # Configuramos la regresión logística con el balanceo indicado
        clasificador = LogisticRegression(
            C=c_param,
            max_iter=max_iter,
            solver=solver,
            class_weight=peso_clases,
            random_state=semilla
        )

        # Si el usuario eligió liblinear, aplicamos One-vs-Rest
        if solver == "liblinear":
            clasificador = OneVsRestClassifier(clasificador)

        # Ensamblamos el pipeline con escalador y modelo
        pipeline = Pipeline([
            ("escalador", StandardScaler()),
            ("clasificador", clasificador)
        ])

        # Entrenar el pipeline
        print(f"[ENTRENAMIENTO] Ajustando modelo (C={c_param}, max_iter={max_iter}, solver='{solver}')...")
        pipeline.fit(X_train, y_train)

        # Generar predicciones sobre el conjunto de prueba
        predicciones = pipeline.predict(X_test)

        # Calcular métricas requeridas
        exactitud = accuracy_score(y_test, predicciones)
        f1_ponderado = f1_score(y_test, predicciones, average="weighted")

        print(f"[MÉTRICAS] Exactitud (Accuracy): {exactitud:.4f} ({exactitud * 100:.2f}%)")
        print(f"[MÉTRICAS] F1-Score ponderado:   {f1_ponderado:.4f} ({f1_ponderado * 100:.2f}%)")

        # Generar la imagen para el artefacto
        ruta_imagen = generar_artefacto_matriz(y_test, predicciones)

        # ---------------------------------------------------------------------
        # REGISTRO EN MLFLOW TRACKING (COMPATIBLE CON MLFLOW 3.X)
        # ---------------------------------------------------------------------
        # A) Registro de hiperparámetros
        mlflow.log_param("c_param", c_param)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("solver", solver)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("random_state", semilla)
        mlflow.log_param("preprocesamiento", "StandardScaler_en_Pipeline")

        # B) Registro de métricas
        mlflow.log_metric("accuracy", exactitud)
        mlflow.log_metric("f1_weighted", f1_ponderado)

        # C) Registro del artefacto visual (cumple con log_artifact)
        mlflow.log_artifact(ruta_imagen, artifact_path="graficos")

        # D) Registro del modelo utilizando 'name' para evitar advertencias
        mlflow.sklearn.log_model(sk_model=pipeline, name="modelo_pipeline")

        print("[MLFLOW] Corrida registrada con éxito y sin advertencias en MLflow.\n")


# -----------------------------------------------------------------------------
# ENTRADA PRINCIPAL Y CONFIGURACIÓN DE ARGPARSE
# -----------------------------------------------------------------------------
def main():
    analizador = argparse.ArgumentParser(
        description="Entrenamiento parametrizado de Regresión Logística con MLFlow."
    )

    analizador.add_argument(
        "--data_path",
        type=str,
        default="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        help="Ruta local o URL hacia el archivo CSV de datos."
    )

    analizador.add_argument(
        "--c_param",
        type=float,
        default=1.0,
        help="Inverso de la fuerza de regularización. Valores menores implican mayor penalización."
    )

    analizador.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Número máximo de iteraciones para que converja el optimizador."
    )

    analizador.add_argument(
    "--solver",
    type=str,
    default="lbfgs",
    choices=["lbfgs", "liblinear", "saga", "newton-cg"],
    help="Algoritmo de optimización numérica (lbfgs, liblinear, saga, newton-cg)."
    )

    analizador.add_argument(
        "--class_weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Ponderación de clases: 'none' (todas pesan igual) o 'balanced' (ajusta pesos por frecuencia inversa)."
    )

    analizador.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Semilla aleatoria para partición y entrenamiento (garantiza reproducibilidad)."
    )

    analizador.add_argument(
        "--experiment_name",
        type=str,
        default="Wine_Quality_Baseline_AlejandroFigueroa",
        help="Nombre descriptivo del experimento en MLFlow."
    )

    argumentos = analizador.parse_args()

    entrenar_y_registrar(
        ruta_datos=argumentos.data_path,
        c_param=argumentos.c_param,
        max_iter=argumentos.max_iter,
        solver=argumentos.solver,
        class_weight=argumentos.class_weight,
        semilla=argumentos.random_state,
        nombre_experimento=argumentos.experiment_name
    )


if __name__ == "__main__":
    main()
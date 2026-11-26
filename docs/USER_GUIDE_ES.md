# Manual de Uso — 103KNN Neural Studio

Guía completa de cada vista, control, métrica y endpoint de **103KNN Neural Studio**.

---

## Barra Superior (Global)

Disponible en toda la aplicación:

- **`🌐 ES` / `🌐 EN`**: Cambia el idioma de toda la interfaz en tiempo real (Español / Inglés).
- **`🌙 Dark Theme` / `☀️ Light Theme`**: Alterna entre tema claro y tema oscuro.
- **`API: localhost:5000 (Active)`**: Indica si el servidor HTTP local de automatización e inferencia está escuchando en el puerto `5000`.
- **`Architecture: ...`**: Muestra la arquitectura activa seleccionada para diseñar y entrenar.
- **Pestañas de Navegación**:
  1. **Architecture Designer** (*Diseñador de Arquitectura*)
  2. **Trainer & Evaluation** (*Entrenamiento y Evaluación*)
  3. **Architecture Comparator** (*Comparador de Arquitecturas*)

---

## 1. Vista: Architecture Designer

Permite diseñar visualmente la topología de la red y configurar sus hiperparámetros antes de entrenar.

### Lienzo Interactivo (`InteractiveNetCanvas`)
- **Topología por capas**: Muestra Entrada (celeste), Ocultas (índigo) y Salida (verde esmeralda) con el conteo de neuronas en la insignia superior.
- **Edición manual de conexiones**: En capas de **24 neuronas o menos**, haz clic en una neurona y luego en otra de la capa adyacente derecha para activar o desactivar esa sinapsis individual.
- **Modo compacto para capas masivas**: Si una capa supera las **24 neuronas** (o una transición supera **600 conexiones**), el lienzo no dibuja miles de elipses ni líneas individuales para mantener un renderizado instantáneo; en su lugar muestra una **tarjeta resumen** con la cantidad exacta de neuronas y un haz representativo de curvas Bézier. *(La topología real en memoria para entrenar se mantiene intacta).*
- **`Restore Connections` / `Restaurar Conexiones`**: Restaura la conectividad completa (*full connectivity*) entre todas las capas.
- **Design Log**: Registro inferior con el historial de cambios de estructura y eventos de construcción.

### Panel Derecho de Configuración
- **`ARCHITECTURE TYPE`**: Selecciona el motor a usar en el entrenamiento:
  - `103KNN Neuromorphic` (SNN nativa en C# puro)
  - `Nowral (Static)`, `LibQuid (Liquid)`, `NewNow (Dynamic)` (`TorchSharp`)
  - `FastTree (GBDT Tabular)`, `FastForest (Random Forest)` (`Microsoft.ML`)
  - `Standard MLP (PyTorch)`, `ResNet-Tabular (Skip)`, `LSTM (Recurrent)` (`TorchSharp`)
- **`Network Props` / `Propiedades Red`**:
  - **Structure**: Neuronas por capa separadas por espacios (ej. `8 32 8 2`). Se aplica automáticamente al escribir.
  - **Activation Function**: Función de activación para las arquitecturas densas (`Tanh`, `ReLU`, `Sigmoid`, etc.).
- **`Training Config` / `Config. Entreno`**:
  - **Learning Rate**: Tasa de aprendizaje ($\eta$, por defecto `0.005`).
  - **Batch Size**: Tamaño de mini-lote (por defecto `64`).
  - **Optimizer**: Optimizador (`Adam`, etc.).
- **`Build & Train →` / `Construir y Entrenar →`**: Construye la red con la topología actual y cambia directamente a la vista **Trainer & Evaluation**.

---

## 2. Vista: Trainer & Evaluation

Permite cargar datos reales o sintéticos, entrenar el modelo seleccionado, observar la curva de error, guardar/cargar modelos y ejecutar predicciones manuales.

### Carga de Datos
- **`Load CSV` / `Cargar CSV`**: Carga un archivo `.csv` numérico. La aplicación detecta columnas de entrada y salida según tu topología (o ajusta la topología si es necesario) y aplica normalización Min-Max $[0, 1]$.
- **`Load Parquet` / `Cargar Parquet`**: Carga datasets tabulares en formato Apache Parquet (`.parquet`), ideal para datasets grandes de Hugging Face o Kaggle sin convertirlos a CSV.

### Entrenamiento y Persistencia
- **`▶ Start` / `▶ Iniciar`**: Inicia el entrenamiento en segundo plano hasta alcanzar el máximo de épocas o la meta de error (`Target Error (MSE)`).
- **`⏹ Stop` / `⏹ Detener`**: Detiene el entrenamiento de forma segura en la época actual.
- **`Reset Weights` / `Reiniciar Pesos`**: Reinicializa los pesos sin perder el dataset cargado.
- **`Save Model` / `Guardar Modelo`**: Exporta el modelo entrenado a `.json` (o `.bin` según arquitectura). En `103KNN Neuromorphic`, guarda los pesos, conectividad dispersa, parámetros LIF/TTFS **y las escalas `MinIn`, `MaxIn`, `MinOut`, `MaxOut`** para usar `PredictRaw(...)` directamente en producción.
- **`Load Model` / `Cargar Modelo`**: Carga un modelo previamente guardado y restaura sus escalas de normalización y topología.

### Visualización e Inferencia Manual
- **Gráfico de Pérdida (`TrainingLossChart`)**: Muestra en tiempo real la evolución del MSE por época.
- **Inferencia Manual**: Escribe un vector de entrada (valores separados por comas o espacios, ej. `1, 0, 1, 0, 1, 1, 0, 0`) y pulsa **`Run Prediction` / `Ejecutar Predicción`** para ver la salida predicha en pantalla.

---

## 3. Vista: Architecture Comparator

Permite enfrentar múltiples arquitecturas bajo exactamente el mismo dataset, partición y presupuesto de épocas/error.

### Configuración (`BENCHMARK SETUP`)
- **`ARCHITECTURES TO EVALUATE`**: Marca las arquitecturas que deseas comparar (puedes incluir las 9 disponibles).
- **`COMMON DATASET & TOPOLOGY`**:
  - Muestra la topología actual y el número de muestras activas.
  - **`Load CSV`**, **`Load Parquet`** o **`Generate Synthetic Dataset (240)`** para generar un dataset senoidal no lineal de prueba al instante.
- **`TEST PARAMETERS`**:
  - **Max Epochs per Model**: Límite de épocas por modelo (en `FastTree`/`FastForest` equivale al número de árboles por salida, p. ej. `200`).
  - **Target Error (MSE)**: Umbral de parada temprana cuando un modelo alcanza ese error.
  - **`80/20 Train/Test Split (Generalization)`**: Divide automáticamente el dataset en 80% entrenamiento y 20% prueba para medir generalización real (`Test MSE`).
- **`▶ Run Benchmark` / `▶ Ejecutar Comparativa`** y **`⏹`**: Ejecuta o cancela la comparativa secuencial.

### Lectura de Resultados
- **Gráfico de Barras (`BenchmarkBarChart`)**: Compara visualmente el tiempo total de entrenamiento (ms) y el MSE alcanzado.
- **Tabla Comparativa**:
  - **Train MSE**: Error cuadrático medio sobre el conjunto de entrenamiento.
  - **Test MSE**: Error sobre el 20% de prueba no visto (si está activo el split 80/20).
  - **Epochs**: Épocas ejecutadas hasta converger o llegar al límite.
  - **Total Time / ms per Epoch**: Tiempo total y promedio por época.
  - **Infer (µs/sample)**: Latencia media de inferencia unitaria ($B = 1$) en microsegundos.
  - **Ops / Spikes**: Operaciones o espigas activas promedio por inferencia (en `103KNN Neuromorphic` refleja la escasez real evento-dirigida).
  - **Result**: Indica qué modelo fue `Fastest`, `Lowest MSE`, `Target reached` o `Completed`.

---

## 4. API HTTP Local (`http://localhost:5000`)

Mientras **103KNN Neural Studio** está abierto, expone una API JSON local en `http://localhost:5000/` para automatización de pruebas o integración con scripts externos:

| Método y Ruta | Descripción | Ejemplo de Body JSON |
|---|---|---|
| `GET /help` | Lista todos los endpoints e IDs de arquitecturas (`0`–`8`) | — |
| `GET /status` | Estado actual (arquitectura, capas, conexiones, muestras) | — |
| `POST /ui/view` | Cambia la pestaña activa | `{"view":"designer\|trainer\|benchmark"}` |
| `POST /design` | Define arquitectura y topología | `{"NetworkType":3,"Inputs":8,"HiddenLayers":[32,8],"Outputs":2}` |
| `POST /build` | Construye la red | — |
| `POST /dataset/load-csv` | Carga un archivo CSV | `{"path":"E:\\GitHub\\103KNN\\datasets\\8 in 2 out.csv"}` |
| `POST /dataset/generate-synthetic` | Genera dataset sintético | `{"sampleCount":240}` |
| `POST /benchmark/run` | Ejecuta el comparador | `{"architectures":[0,2,3,4,5],"maxEpochs":2000,"targetError":0.0001,"learningRate":0.005,"batchSize":64,"trainTestSplit":true}` |
| `GET /benchmark/results` | Devuelve los resultados de la última comparativa | — |
| `POST /train/start` | Prepara el entrenador con la red actual | — |
| `POST /train/load-data` | Carga tensores directamente | `{"Inputs":[[...]],"Targets":[[...]]}` |
| `POST /train/execute` | Ejecuta el entrenamiento | — |
| `GET /train/progress` | Consulta progreso y MSE actual | — |
| `POST /train/stop` | Detiene el entrenamiento | — |
| `POST /predict` | Ejecuta inferencia unitaria | `[0.5, 0.5]` |

---

## 5. Uso en Producción con NuGet (`103KNN.Neuromorphic`)

Si solo deseas usar el motor neuronal neuromórfico en tus propias aplicaciones .NET (sin la interfaz WPF ni dependencias de `TorchSharp`/`ML.NET`), instala el paquete ligero (~87 KB):

```powershell
dotnet add package 103KNN.Neuromorphic
```

Consulta el ejemplo completo en C# (entrenamiento, guardado/carga con escalas Min-Max e inferencia `PredictRaw`) en el [README.md](../README.md#quick-start-using-103knnneuromorphic-from-c-code).

# User Guide — 103KNN Neural Studio

Complete guide to every view, control, metric, and HTTP endpoint in **103KNN Neural Studio**.

---

## Top Header Bar (Global)

Available across all views:

- **`🌐 ES` / `🌐 EN`**: Switches the entire UI language in real time (English / Spanish).
- **`🌙 Dark Theme` / `☀️ Light Theme`**: Toggles between Light and Dark themes.
- **`API: localhost:5000 (Active)`**: Shows whether the local HTTP server for automation and inference is listening on port `5000`.
- **`Architecture: ...`**: Displays the currently selected architecture for design and training.
- **Navigation Tabs**:
  1. **Architecture Designer**
  2. **Trainer & Evaluation**
  3. **Architecture Comparator**

---

## 1. View: Architecture Designer

Lets you visually design the network topology and configure hyperparameters before training.

### Interactive Canvas (`InteractiveNetCanvas`)
- **Layer visualization**: Displays Input (sky cyan), Hidden (indigo), and Output (emerald) layers with neuron counts in the top badge.
- **Manual connection editing**: For layers with **24 neurons or fewer**, click one neuron and then a neuron in the adjacent right layer to toggle that individual synapse on/off.
- **Compact mode for massive layers**: When a layer exceeds **24 neurons** (or a layer-to-layer map exceeds **600 connections**), the canvas skips drawing thousands of individual ellipses/curves to keep rendering instantaneous, displaying instead a **summary card** with the exact neuron count and a representative bundle of Bézier curves. *(The underlying `ConnectionMap` used for training remains exact.)*
- **`Restore Connections`**: Restores full connectivity across all layers.
- **Design Log**: Bottom console showing topology changes and build events.

### Right Configuration Panel
- **`ARCHITECTURE TYPE`**: Selects the engine used for training:
  - `103KNN Neuromorphic` (Pure-C# native SNN)
  - `Nowral (Static)`, `LibQuid (Liquid)`, `NewNow (Dynamic)` (`TorchSharp`)
  - `FastTree (GBDT Tabular)`, `FastForest (Random Forest)` (`Microsoft.ML`)
  - `Standard MLP (PyTorch)`, `ResNet-Tabular (Skip)`, `LSTM (Recurrent)` (`TorchSharp`)
- **`Network Props`**:
  - **Structure**: Space-separated neuron counts per layer (e.g. `8 32 8 2`). Auto-applies as you type.
  - **Activation Function**: Activation function for dense neural architectures (`Tanh`, `ReLU`, `Sigmoid`, etc.).
- **`Training Config`**:
  - **Learning Rate**: Step size ($\eta$, default `0.005`).
  - **Batch Size**: Mini-batch size (default `64`).
  - **Optimizer**: Optimizer selection (`Adam`, etc.).
- **`Build & Train →`**: Builds the network with the current topology and switches directly to **Trainer & Evaluation**.

---

## 2. View: Trainer & Evaluation

Lets you load real or synthetic datasets, train the selected model, inspect the live loss curve, save/load models, and run manual predictions.

### Dataset Loading
- **`Load CSV`**: Loads a numeric `.csv` file, infers/adapts input and output dimensions, and applies Min-Max $[0, 1]$ normalization.
- **`Load Parquet`**: Loads Apache Parquet (`.parquet`) tabular datasets directly (ideal for modern Hugging Face or Kaggle datasets without converting to CSV).

### Training & Persistence
- **`▶ Start`**: Starts background training until reaching max epochs or `Target Error (MSE)`.
- **`⏹ Stop`**: Gracefully stops training at the current epoch.
- **`Reset Weights`**: Reinitializes weights while keeping the loaded dataset.
- **`Save Model`**: Exports the trained model to `.json` (or `.bin` depending on architecture). For `103KNN Neuromorphic`, it saves weights, sparse connectivity, LIF/TTFS parameters, **and `MinIn`, `MaxIn`, `MinOut`, `MaxOut` normalization scales** so `PredictRaw(...)` works plug-and-play in production.
- **`Load Model`**: Loads a previously saved model and restores its topology and normalization scales.

### Visualization & Manual Inference
- **Loss Chart (`TrainingLossChart`)**: Plots MSE convergence per epoch in real time.
- **Manual Inference**: Enter an input vector (comma- or space-separated values, e.g. `1, 0, 1, 0, 1, 1, 0, 0`) and click **`Run Prediction`** to view the predicted output.

---

## 3. View: Architecture Comparator

Benchmarks multiple architectures side by side under the exact same dataset, train/test split, and epoch/error budget.

### Setup (`BENCHMARK SETUP`)
- **`ARCHITECTURES TO EVALUATE`**: Checkboxes to select any subset of the 9 supported architectures.
- **`COMMON DATASET & TOPOLOGY`**:
  - Shows active topology and sample count.
  - **`Load CSV`**, **`Load Parquet`**, or **`Generate Synthetic Dataset (240)`** to generate a nonlinear sinusoidal test dataset on the fly.
- **`TEST PARAMETERS`**:
  - **Max Epochs per Model**: Epoch limit per model (for `FastTree`/`FastForest`, sets the number of trees per output, e.g. `200`).
  - **Target Error (MSE)**: Early-stopping target MSE.
  - **`80/20 Train/Test Split (Generalization)`**: Splits the dataset into 80% training and 20% unseen test data to measure true generalization (`Test MSE`).
- **`▶ Run Benchmark`** / **`⏹`**: Starts or cancels the sequential benchmark.

### Reading Results
- **Bar Chart (`BenchmarkBarChart`)**: Visual comparison of total training time (ms) and final MSE.
- **Metrics Table**:
  - **Train MSE**: Mean Squared Error on the training split.
  - **Test MSE**: Mean Squared Error on the 20% unseen test split.
  - **Epochs**: Epochs run until convergence or limit.
  - **Total Time / ms per Epoch**: Total training duration and average per epoch.
  - **Infer (µs/sample)**: Average single-sample ($B = 1$) inference latency in microseconds.
  - **Ops / Spikes**: Average active operations or spikes per inference (for `103KNN Neuromorphic`, reflects actual event-driven sparsity).
  - **Result**: Highlights `Fastest`, `Lowest MSE`, `Target reached`, or `Completed`.

---

## 4. Local HTTP API (`http://localhost:5000`)

While **103KNN Neural Studio** is running, it exposes a local JSON API on `http://localhost:5000/` for automated testing and external scripts:

| Method & Path | Description | Example JSON Body |
|---|---|---|
| `GET /help` | Lists all endpoints and architecture IDs (`0`–`8`) | — |
| `GET /status` | Current status (architecture, layers, connections, samples) | — |
| `POST /ui/view` | Switches active tab | `{"view":"designer\|trainer\|benchmark"}` |
| `POST /design` | Sets architecture and topology | `{"NetworkType":3,"Inputs":8,"HiddenLayers":[32,8],"Outputs":2}` |
| `POST /build` | Builds the network | — |
| `POST /dataset/load-csv` | Loads a CSV file | `{"path":"E:\\GitHub\\103KNN\\datasets\\8 in 2 out.csv"}` |
| `POST /dataset/generate-synthetic` | Generates synthetic dataset | `{"sampleCount":240}` |
| `POST /benchmark/run` | Runs the comparator | `{"architectures":[0,2,3,4,5],"maxEpochs":2000,"targetError":0.0001,"learningRate":0.005,"batchSize":64,"trainTestSplit":true}` |
| `GET /benchmark/results` | Returns latest benchmark results | — |
| `POST /train/start` | Prepares trainer with current network | — |
| `POST /train/load-data` | Loads raw input/target arrays | `{"Inputs":[[...]],"Targets":[[...]]}` |
| `POST /train/execute` | Executes training | — |
| `GET /train/progress` | Queries training progress & MSE | — |
| `POST /train/stop` | Stops training | — |
| `POST /predict` | Runs single-sample inference | `[0.5, 0.5]` |

---

## 5. Production Use via NuGet (`103KNN.Neuromorphic`)

If you only need the neuromorphic engine in your own .NET applications (without the WPF Studio or `TorchSharp`/`ML.NET` dependencies), install the lightweight (~87 KB) package:

```powershell
dotnet add package 103KNN.Neuromorphic
```

See the full C# code example (training, saving/loading with Min-Max scales, and `PredictRaw` inference) in [README.md](../README.md#quick-start-using-103knnneuromorphic-from-c-code).

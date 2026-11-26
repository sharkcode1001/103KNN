# 103KNN Neural Studio

**A native .NET 10 Neuromorphic Computing & Multi-Architecture Neural Network Studio.**

`103KNN` combines a **100% pure C# event-driven Spiking Neural Network (SNN) engine** (`103KNN.Neuromorphic`) with an interactive WPF studio for designing sparse topologies, training models on CSV/Parquet datasets, and benchmarking neuromorphic architectures side-by-side against `TorchSharp`-based static, dynamic, and liquid neural networks.

---

## Key Highlights

- **Pure C# Neuromorphic Engine (`103KNN.Neuromorphic`):**
  - Zero third-party dependencies — runs natively on CPU without Python or `libtorch`.
  - **Time-to-First-Spike (TTFS)** temporal encoding & **Leaky Integrate-and-Fire (LIF)** neurons.
  - Hybrid **Surrogate-Gradient + Causal STDP + Adam** learning rule.
  - **Sparse Adjacency & Multi-Core CPU Parallelism** (`Parallel.For` with thread-local scratchpads), eliminating dense matrix overhead and native FFI marshaling.
- **Interactive Architecture Designer:**
  - Visual topology canvas with smooth Bezier connections; click any two adjacent neurons to enable or disable individual synapses.
  - Customize layer structure, activation functions (`ReLU`, `Sigmoid`, `LeakyReLU`), learning rate, batch size, optimizer (`Adam`, `SGD`, `RMSprop`), and STDP rate.
- **Multi-Architecture Trainer & Live Evaluation:**
  - Supports **103KNN Neuromorphic**, **Nowral (Static)**, **NewNow (Dynamic)**, and **LibQuid (Liquid ODE)**.
  - Real-time MSE convergence chart, CSV & Parquet data loaders with automatic Min-Max normalization/denormalization, model persistence (`.json` / `.weights.bin`), and live manual inference.
- **Side-by-Side Architecture Comparator:**
  - Evaluates multiple architectures on the exact same topology and dataset, measuring **Final MSE**, **Epochs**, **Total Training Time**, **ms/Epoch**, **Single-Sample Inference Latency (µs/sample)**, and **Ops / Spikes per sample**.
- **Modern Bilingual UI & Local HTTP API:**
  - Instant **Light / Dark Theme** toggle and **English / Spanish** switcher.
  - Built-in local HTTP API (`http://localhost:5000/`) for remote design, training control, progress polling, and prediction (`/status`, `/design`, `/build`, `/train/start`, `/train/load-data`, `/train/execute`, `/train/progress`, `/train/stop`, `/predict`).

---

## Observed Benchmark Example

Tested on a non-linear 8-input, 4-output sinusoidal regression dataset (`240` samples, topology `8-32-8-4`, `544` connections, `LR = 0.005`, target `MSE <= 0.0001`, max `4000` epochs):

| Architecture | Final MSE | Epochs | Total Time | ms / Epoch | Inference (µs/sample) | Ops / Spikes | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| **103KNN Neuromorphic** | **0.000099** | **2,324** | 11,254.1 ms | 4.84 ms | **23.6 µs** | **35** | **Lowest MSE** |
| **NewNow (Dynamic)** | 0.000418 | 4,000 | 19,994.4 ms | 5.00 ms | 117.5 µs | 544 | Completed |
| **Nowral (Static)** | 0.002571 | 4,000 | 5,082.1 ms | 1.27 ms | 173.0 µs | 544 | Fastest (did not reach target) |

- **7.3× faster single-sample inference** than `Nowral` and **5.0× faster** than `NewNow` (~42,300 inferences/sec on CPU).
- **15.5× fewer operations** per sample (`35` sparse spike events vs. `544` dense operations).
- **Only architecture** to reach the target `MSE <= 0.0001` on this non-linear task.

---

## Solution Structure

- `103KNN.Neuromorphic/` — Core pure-C# neuromorphic engine (`NeuromorphicTrainableNetwork`, `NeuromorphicModelPersistence`, `CognitiveLifNeuron`, `TimeToFirstSpikeEncoder`, `StdpSynapticTrainer`, `NeuromorphicReasoner`, `NeuromorphicKnnEngine`).
- `103KNN/` — WPF Studio application (`MainWindow`, `InteractiveNetCanvas`, `TrainingLossChart`, `BenchmarkBarChart`, `ModelTrainerEngine`, `TabularTreeEnsemble`, `NeuralApiServer`).
- `datasets/` — Sample & public UCI/Kaggle CSV datasets ready for training and 80/20 Train/Test benchmarking.
- `docs/` — Scientific evaluation papers in [English](docs/PAPER_EN.md) and [Spanish](docs/PAPER_ES.md).
- `NowLib/` — Shared topology (`ConnectionMap`) and activation interfaces (`IActivation`).
- `Nowral/` — Static masked feedforward network (`TorchSharp`).
- `NewNow/` — Dynamic mini-batch masked network (`TorchSharp`).
- `LibQuid/` — Liquid time-constant ODE network (`TorchSharp`).

---

## Sample Datasets (`datasets/`) & How It Works

The `datasets/` folder includes 6 ready-to-use `.csv` files for testing both static/tabular and sequential (`LibQuid`) architectures:

| File | Rows | Inputs → Outputs | Suggested Structure | Description |
|---|---:|---|---|---|
| `2 in 1 out.csv` | 4 | `2 → 1` | `2 8 1` | 2-input binary truth table |
| `2 in 1 out liquid.csv` | 9 | `2 → 1` | `2 8 1` | Sequential ramp for temporal / liquid testing |
| `4 in 2 out.csv` | 16 | `4 → 2` | `4 16 8 2` | 4-bit binary combinations (`2⁴` rows) |
| `4 in 2 out - liquid.csv` | 401 | `4 → 2` | `4 16 8 2` | Continuous 401-step time series (ideal for `LibQuid`) |
| `7 in 3 out.csv` | 128 | `7 → 3` | `7 24 8 3` | 7-bit binary combinations (`2⁷` rows) |
| `8 in 2 out.csv` | 256 | `8 → 2` | `8 32 8 2` | 8-bit binary combinations (`2⁸` rows) |

### How CSV Loading & Training Works
1. **Set the matching topology:** In **Architecture Designer**, enter a structure whose first number matches the dataset's input columns and last number matches its output columns (for example, `8 32 8 2` for `8 in 2 out.csv`) and click **Apply Structure**.
2. **Load the dataset:** In **Trainer & Evaluation** or **Architecture Comparator**, click **Load CSV** and select a file from `datasets/`.
   - Each row is parsed automatically (comma, semicolon, or tab separated; optional non-numeric header row is skipped).
   - The first `InputCount` columns are used as inputs and the remaining `OutputCount` columns as targets.
   - All inputs and targets are **Min-Max normalized to `[0, 1]`** during training, and **automatically denormalized** back to the original scale when running **Live Inference** (`Run Prediction`).
3. **Train or Benchmark:** Start training to watch the real-time MSE curve, or run **Architecture Comparator** to compare **103KNN Neuromorphic**, **Nowral**, **NewNow**, and **LibQuid** on the exact same dataset.

---

## Requirements & Build

- **OS:** Windows (x64)
- **SDK:** [.NET 10 SDK](https://dotnet.microsoft.com/)

```bash
git clone https://github.com/<your-username>/103KNN.git
cd 103KNN
dotnet build 103KNN.slnx
dotnet run --project 103KNN/103KNN.csproj
```

---

## Licensing

`103KNN` and `103KNN.Neuromorphic` are released under a **Dual-Licensing Model** (see [LICENSE](LICENSE) and [LICENSING.md](LICENSING.md)):

1. **Open Source & Academic Use:** Licensed under the **GNU General Public License v3.0 (GPLv3)**.
2. **Commercial & Closed-Source Use:** A **Private Commercial License** from the author is required for proprietary or closed-source products.

---

## Contact

- **Author:** Erisnel Lora Sugve  
- **Email:** `lorasugve@gmail.com`  
- **Location:** Santiago de Cuba, Cuba

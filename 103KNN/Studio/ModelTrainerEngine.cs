using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Training;
using LibQuid;
using NewNow;
using NowLib;
using Nowral;
using ParquetSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace _103KNN.Studio;

/// <summary>
/// Motor unificado de entrenamiento, inferencia y persistencia para las 4 arquitecturas:
/// Nowral (Estática), LibQuid (Líquida), NewNow (Dinámica) y 103KNN Neuromorphic (SNN).
/// </summary>
public sealed class ModelTrainerEngine : IDisposable
{
    private const int LiquidWindowSize = 5;

    private NowralNetwork? _nowralNet;
    private LibQuidNetwork? _libQuidNet;
    private NewNowNetwork? _newNowNet;
    private NeuromorphicTrainableNetwork? _neuroNet;
    private TabularTreeEnsemble? _treeEnsemble;
    private StandardMlpNetwork? _stdMlpNet;
    private ResNetTabularNetwork? _resNetNet;
    private LstmTabularNetwork? _lstmNet;

    private torch.optim.Optimizer? _torchOptimizer;
    private CancellationTokenSource? _cts;

    private float[][]? _rawInputs;
    private float[][]? _rawTargets;
    private float[][]? _normInputs;
    private float[][]? _normTargets;
    private float[,,]? _liquidInputs;
    private float[][]? _liquidTargets;

    private float[]? _minIn, _maxIn;
    private float[]? _minOut, _maxOut;
    private readonly Queue<float[]> _liquidPredictionBuffer = new();
    private readonly Queue<float[]> _lstmPredictionBuffer = new();

    public NetworkArchitectureType Architecture { get; private set; }
    public List<ConnectionMap> Maps { get; private set; } = new();
    public IActivation Activation { get; private set; } = new LeakyReLUActivation();
    public TrainingSettings Settings { get; private set; } = new();

    public bool IsTraining { get; private set; }
    public long CurrentEpoch { get; private set; }
    public float CurrentMSE { get; private set; }
    public long LastOpsOrSpikes { get; private set; }
    public long LastLifFiredCount { get; private set; }
    public long LastPropagatedEventCount { get; private set; }
    public long LastSynapticOpsCount { get; private set; }
    public int SampleCount => _rawInputs?.Length ?? 0;
    public float[][]? RawInputs => _rawInputs;
    public float[][]? RawTargets => _rawTargets;

    public event Action<string>? OnLog;
    public event Action? OnProgressUpdated;
    public event Action<bool>? OnTrainingStateChanged;
    public event Action<List<ConnectionMap>>? OnTopologyChanged;

    public void Initialize(
        NetworkArchitectureType architecture,
        List<ConnectionMap> maps,
        IActivation activation,
        TrainingSettings settings)
    {
        Stop();
        DisposeNetworks();

        Architecture = architecture;
        Maps = maps.Select(CloneMap).ToList();
        Activation = activation;
        Settings = settings;
        CurrentEpoch = 0;
        CurrentMSE = 0f;
        LastOpsOrSpikes = 0;
        _liquidPredictionBuffer.Clear();
        _lstmPredictionBuffer.Clear();

        if (Maps.Count == 0) return;

        switch (Architecture)
        {
            case NetworkArchitectureType.Nowral:
                _nowralNet = new NowralNetwork(Maps, Activation);
                _torchOptimizer = CreateTorchOptimizer(_nowralNet.parameters());
                LastOpsOrSpikes = Maps.Sum(m => (long)m.Connections.Count);
                break;

            case NetworkArchitectureType.LibQuid:
                _libQuidNet = new LibQuidNetwork(Maps);
                _libQuidNet.to(ScalarType.Float32);
                _torchOptimizer = CreateTorchOptimizer(_libQuidNet.parameters());
                LastOpsOrSpikes = Maps.Sum(m => (long)m.SourceSize * m.TargetSize) * LiquidWindowSize;
                PrepareLiquidSequences();
                break;

            case NetworkArchitectureType.NewNow:
                _newNowNet = new NewNowNetwork(Maps, Activation);
                _torchOptimizer = CreateTorchOptimizer(_newNowNet.parameters());
                LastOpsOrSpikes = Maps.Sum(m => (long)m.Connections.Count);
                break;

            case NetworkArchitectureType.Neuromorphic:
                var neuroMaps = ToNeuromorphicMaps(Maps);
                _neuroNet = new NeuromorphicTrainableNetwork(neuroMaps, new CognitiveConfig
                {
                    BaseThreshold = 0.25f,
                    MembraneDecay = 0.90f,
                    LateralInhibitionFactor = 0.05f,
                    ThresholdAdaptationRate = 0.02f,
                    TimeWindow = 1.0f
                });
                LastOpsOrSpikes = 0;
                break;

            case NetworkArchitectureType.FastTree:
            case NetworkArchitectureType.FastForest:
                _treeEnsemble = new TabularTreeEnsemble(
                    Maps[0].SourceSize,
                    Maps[^1].TargetSize,
                    isRandomForest: Architecture == NetworkArchitectureType.FastForest,
                    numberOfTrees: 200);
                LastOpsOrSpikes = _treeEnsemble.NumberOfTrees * Maps[^1].TargetSize;
                break;

            case NetworkArchitectureType.StandardMlp:
                _stdMlpNet = new StandardMlpNetwork(GetLayerSizes(), Activation);
                _torchOptimizer = CreateTorchOptimizer(_stdMlpNet.parameters());
                LastOpsOrSpikes = Maps.Sum(m => (long)m.SourceSize * m.TargetSize);
                break;

            case NetworkArchitectureType.ResNetTabular:
                int hiddenW = Maps.Count > 1 ? Maps[0].TargetSize : 32;
                _resNetNet = new ResNetTabularNetwork(Maps[0].SourceSize, hiddenW, Maps[^1].TargetSize, numBlocks: 2);
                _torchOptimizer = CreateTorchOptimizer(_resNetNet.parameters());
                LastOpsOrSpikes = Maps[0].SourceSize * hiddenW + (4L * hiddenW * hiddenW) + hiddenW * Maps[^1].TargetSize;
                break;

            case NetworkArchitectureType.Lstm:
                int lstmH = Maps.Count > 1 ? Maps[0].TargetSize : 32;
                _lstmNet = new LstmTabularNetwork(Maps[0].SourceSize, lstmH, Maps[^1].TargetSize);
                _torchOptimizer = CreateTorchOptimizer(_lstmNet.parameters());
                LastOpsOrSpikes = (4L * (Maps[0].SourceSize + lstmH) * lstmH) * LiquidWindowSize;
                PrepareLiquidSequences();
                break;
        }

        OnLog?.Invoke($"Red [{Architecture}] inicializada. Capas: {string.Join("-", GetLayerSizes())}");
        OnProgressUpdated?.Invoke();
    }

    public int[] GetLayerSizes()
    {
        if (Maps.Count == 0) return Array.Empty<int>();
        var list = new List<int> { Maps[0].SourceSize };
        list.AddRange(Maps.Select(m => m.TargetSize));
        return list.ToArray();
    }

    public void SetData(float[][] inputs, float[][] targets, bool normalize = true)
    {
        if (inputs.Length == 0 || targets.Length == 0) return;

        _rawInputs = inputs;
        _rawTargets = targets;

        if (normalize)
        {
            _normInputs = Normalize(inputs, out _minIn, out _maxIn);
            _normTargets = Normalize(targets, out _minOut, out _maxOut);
        }
        else
        {
            _normInputs = inputs;
            _normTargets = targets;
            _minIn = _maxIn = _minOut = _maxOut = null;
        }

        if (Architecture == NetworkArchitectureType.LibQuid || Architecture == NetworkArchitectureType.Lstm)
        {
            PrepareLiquidSequences();
        }

        OnLog?.Invoke($"Dataset cargado: {inputs.Length} muestras ({(normalize ? "Normalizado Min-Max" : "Sin normalizar")}).");
        OnProgressUpdated?.Invoke();
    }

    public void LoadCsv(string filePath)
    {
        if (Maps.Count == 0) throw new InvalidOperationException("Define la estructura de red antes de cargar datos.");

        var lines = File.ReadAllLines(filePath).Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
        if (lines.Length == 0) throw new InvalidOperationException("El archivo CSV está vacío.");

        int inSize = Maps[0].SourceSize;
        int outSize = Maps[^1].TargetSize;
        int expected = inSize + outSize;

        var inputs = new List<float[]>();
        var targets = new List<float[]>();

        foreach (var line in lines)
        {
            var parts = line.Split(new[] { ',', ';', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(p => p.Trim()).ToArray();

            // Saltar cabecera si no es numérica
            if (inputs.Count == 0 && !float.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out _))
                continue;

            if (parts.Length != expected)
                throw new InvalidOperationException($"Cada fila debe tener {expected} valores (inputs+outputs). Encontrado: {parts.Length}");

            var inp = new float[inSize];
            var outp = new float[outSize];
            for (int i = 0; i < inSize; i++) inp[i] = float.Parse(parts[i], CultureInfo.InvariantCulture);
            for (int j = 0; j < outSize; j++) outp[j] = float.Parse(parts[inSize + j], CultureInfo.InvariantCulture);

            inputs.Add(inp);
            targets.Add(outp);
        }

        SetData(inputs.ToArray(), targets.ToArray(), normalize: true);
    }

    public void LoadParquet(string filePath)
    {
        if (Maps.Count == 0) throw new InvalidOperationException("Define la estructura de red antes de cargar datos.");

        using var fileReader = new ParquetFileReader(filePath);
        using var fileMetadata = fileReader.FileMetaData;
        long numRows = fileMetadata.NumRows;
        int numColumns = fileMetadata.NumSchemaElements - 1;

        int inSize = Maps[0].SourceSize;
        int outSize = Maps[^1].TargetSize;

        if (numColumns != inSize + outSize)
            throw new InvalidOperationException($"Columnas en Parquet ({numColumns}) no coinciden con la red ({inSize + outSize}).");

        var rawInputs = new float[numRows][];
        var rawTargets = new float[numRows][];
        for (int i = 0; i < numRows; i++)
        {
            rawInputs[i] = new float[inSize];
            rawTargets[i] = new float[outSize];
        }

        using var rowGroupReader = fileReader.RowGroup(0);
        for (int colIdx = 0; colIdx < numColumns; colIdx++)
        {
            using var columnReader = rowGroupReader.Column(colIdx).LogicalReader<float>();
            var columnData = columnReader.ReadAll((int)numRows);
            for (int r = 0; r < numRows; r++)
            {
                if (colIdx < inSize) rawInputs[r][colIdx] = columnData[r];
                else rawTargets[r][colIdx - inSize] = columnData[r];
            }
        }

        SetData(rawInputs, rawTargets, normalize: true);
        LogNormalizationReport(inSize, outSize);
    }

    public async Task StartTrainingAsync(float targetError, long maxEpochs = 0)
    {
        if (IsTraining) return;
        if (_normInputs == null || _normTargets == null)
        {
            OnLog?.Invoke("⚠️ Carga un dataset (CSV o Parquet) antes de entrenar.");
            return;
        }

        IsTraining = true;
        _cts = new CancellationTokenSource();
        OnTrainingStateChanged?.Invoke(true);
        OnLog?.Invoke($"🚀 Entrenamiento [{Architecture}] iniciado (Objetivo MSE <= {targetError:F6})...");

        try
        {
            await Task.Run(() => RunTrainingLoop(targetError, maxEpochs, _cts.Token));
            if (_cts.Token.IsCancellationRequested)
                OnLog?.Invoke($"⏹️ Entrenamiento detenido por el usuario en época {CurrentEpoch}.");
            else
                OnLog?.Invoke($"🏆 Entrenamiento finalizado. Época {CurrentEpoch} | MSE: {CurrentMSE:F8}");
        }
        catch (Exception ex)
        {
            OnLog?.Invoke($"❌ Error en entrenamiento: {ex.Message}");
        }
        finally
        {
            IsTraining = false;
            OnTrainingStateChanged?.Invoke(false);
            OnProgressUpdated?.Invoke();
        }
    }

    public void Stop()
    {
        try { _cts?.Cancel(); } catch (ObjectDisposedException) { }
    }

    /// <summary>
    /// Evalúa el MSE sobre un conjunto de prueba en la misma escala normalizada [0, 1] del entrenamiento,
    /// sin fuga de datos (usa _minIn/_maxIn/_minOut/_maxOut calculados en SetData).
    /// </summary>
    public float EvaluateNormalizedMse(float[][] testInputs, float[][] testTargets)
    {
        if (testInputs.Length == 0 || testTargets.Length == 0 || Maps.Count == 0) return float.NaN;

        _liquidPredictionBuffer.Clear();
        _lstmPredictionBuffer.Clear();
        double totalSqErr = 0;
        int evaluated = 0;
        int outDim = Maps[^1].TargetSize;

        for (int i = 0; i < testInputs.Length; i++)
        {
            var res = Predict(testInputs[i]);
            if (!res.Success || res.Prediction == null) continue;

            for (int k = 0; k < outDim && k < res.Prediction.Length; k++)
            {
                float predNorm = res.Prediction[k];
                float targNorm = testTargets[i][k];
                if (_minOut != null && _maxOut != null && k < _minOut.Length)
                {
                    float range = _maxOut[k] - _minOut[k];
                    predNorm = range == 0 ? 0f : (predNorm - _minOut[k]) / range;
                    targNorm = range == 0 ? 0f : (targNorm - _minOut[k]) / range;
                }
                double err = predNorm - targNorm;
                totalSqErr += err * err;
            }
            evaluated++;
        }

        return evaluated > 0 ? (float)(totalSqErr / (evaluated * outDim)) : float.NaN;
    }

    public float[] NormalizeInput(float[] rawInput)
    {
        var input = (float[])rawInput.Clone();
        if (_minIn != null && _maxIn != null)
        {
            for (int i = 0; i < input.Length && i < _minIn.Length; i++)
            {
                float range = _maxIn[i] - _minIn[i];
                input[i] = range == 0 ? 0f : (input[i] - _minIn[i]) / range;
            }
        }
        return input;
    }

    public bool PredictNormalized(float[] normalizedInput)
    {
        if (Maps.Count == 0 || normalizedInput.Length != Maps[0].SourceSize) return false;
        return PredictNormalizedCore(normalizedInput, notifyProgress: false) != null;
    }

    public PredictionResultDto Predict(float[] rawInput)
    {
        try
        {
            if (Maps.Count == 0) return PredictionResultDto.Fail("No hay red cargada.");
            int inSize = Maps[0].SourceSize;
            if (rawInput.Length != inSize)
                return PredictionResultDto.Fail($"Se esperaban {inSize} valores, se recibieron {rawInput.Length}.");

            var input = NormalizeInput(rawInput);
            float[]? pred = PredictNormalizedCore(input, notifyProgress: true, out string? errorMsg);
            if (pred == null)
                return PredictionResultDto.Fail(errorMsg ?? "Error en predicción.");

            if (_minOut != null && _maxOut != null)
            {
                for (int i = 0; i < pred.Length && i < _minOut.Length; i++)
                {
                    float range = _maxOut[i] - _minOut[i];
                    pred[i] = (pred[i] * range) + _minOut[i];
                }
            }

            return PredictionResultDto.Ok(pred);
        }
        catch (Exception ex)
        {
            return PredictionResultDto.Fail(ex.Message);
        }
    }

    private float[]? PredictNormalizedCore(float[] input, bool notifyProgress) =>
        PredictNormalizedCore(input, notifyProgress, out _);

    private float[]? PredictNormalizedCore(float[] input, bool notifyProgress, out string? errorMsg)
    {
        errorMsg = null;
        int inSize = Maps[0].SourceSize;
        float[] pred;
        switch (Architecture)
        {
            case NetworkArchitectureType.Nowral:
                if (_nowralNet == null) { errorMsg = "Red Nowral no inicializada."; return null; }
                _nowralNet.eval();
                using (var tIn = torch.tensor(input, new long[] { 1, input.Length }, dtype: ScalarType.Float32))
                using (var tOut = _nowralNet.forward(tIn))
                    pred = tOut.data<float>().ToArray();
                break;

            case NetworkArchitectureType.NewNow:
                if (_newNowNet == null) { errorMsg = "Red NewNow no inicializada."; return null; }
                _newNowNet.eval();
                using (var scope = torch.NewDisposeScope())
                {
                    var tIn = torch.tensor(input, ScalarType.Float32).view(1, -1);
                    var tOut = _newNowNet.forward(tIn);
                    pred = tOut.data<float>().ToArray();
                }
                break;

            case NetworkArchitectureType.LibQuid:
                if (_libQuidNet == null) { errorMsg = "Red LibQuid no inicializada."; return null; }
                _liquidPredictionBuffer.Enqueue(input);
                if (_liquidPredictionBuffer.Count > LiquidWindowSize) _liquidPredictionBuffer.Dequeue();
                if (_liquidPredictionBuffer.Count < LiquidWindowSize)
                {
                    errorMsg = $"Buffer temporal: {_liquidPredictionBuffer.Count}/{LiquidWindowSize} pasos acumulados.";
                    return null;
                }

                var seqData = _liquidPredictionBuffer.SelectMany(x => x).ToArray();
                using (var scope = torch.NewDisposeScope())
                {
                    using var tIn = torch.tensor(seqData, new long[] { 1, LiquidWindowSize, inSize }).to(ScalarType.Float32);
                    using var tOut = _libQuidNet.forward(tIn);
                    pred = tOut.data<float>().ToArray();
                }
                break;

            case NetworkArchitectureType.Neuromorphic:
                if (_neuroNet == null) { errorMsg = "Red Neuromórfica no inicializada."; return null; }
                pred = _neuroNet.Predict(input, out var neuroMetrics);
                LastOpsOrSpikes = _neuroNet.LastSpikesProcessed;
                LastLifFiredCount = neuroMetrics.LifFiredCount;
                LastPropagatedEventCount = neuroMetrics.PropagatedEventCount;
                LastSynapticOpsCount = neuroMetrics.SynapticOpsCount;
                if (notifyProgress) OnProgressUpdated?.Invoke();
                break;

            case NetworkArchitectureType.FastTree:
            case NetworkArchitectureType.FastForest:
                if (_treeEnsemble == null) { errorMsg = "Modelo tabular no inicializado."; return null; }
                pred = _treeEnsemble.Predict(input);
                break;

            case NetworkArchitectureType.StandardMlp:
                if (_stdMlpNet == null) { errorMsg = "Red Standard MLP no inicializada."; return null; }
                _stdMlpNet.eval();
                using (var scope = torch.NewDisposeScope())
                {
                    var tIn = torch.tensor(input, ScalarType.Float32).view(1, -1);
                    pred = _stdMlpNet.forward(tIn).data<float>().ToArray();
                }
                break;

            case NetworkArchitectureType.ResNetTabular:
                if (_resNetNet == null) { errorMsg = "Red ResNet-Tabular no inicializada."; return null; }
                _resNetNet.eval();
                using (var scope = torch.NewDisposeScope())
                {
                    var tIn = torch.tensor(input, ScalarType.Float32).view(1, -1);
                    pred = _resNetNet.forward(tIn).data<float>().ToArray();
                }
                break;

            case NetworkArchitectureType.Lstm:
                if (_lstmNet == null) { errorMsg = "Red LSTM no inicializada."; return null; }
                _lstmPredictionBuffer.Enqueue(input);
                if (_lstmPredictionBuffer.Count > LiquidWindowSize) _lstmPredictionBuffer.Dequeue();
                if (_lstmPredictionBuffer.Count < LiquidWindowSize)
                {
                    errorMsg = $"Buffer LSTM: {_lstmPredictionBuffer.Count}/{LiquidWindowSize} pasos.";
                    return null;
                }
                var lstmSeq = _lstmPredictionBuffer.SelectMany(x => x).ToArray();
                _lstmNet.eval();
                using (var scope = torch.NewDisposeScope())
                {
                    var tIn = torch.tensor(lstmSeq, new long[] { 1, LiquidWindowSize, inSize }).to(ScalarType.Float32);
                    pred = _lstmNet.forward(tIn).data<float>().ToArray();
                }
                break;

            default:
                errorMsg = "Arquitectura desconocida.";
                return null;
        }
        return pred;
    }

    public void SaveModel(string basePath)
    {
        switch (Architecture)
        {
            case NetworkArchitectureType.Nowral when _nowralNet != null:
                NowralPersistence.SaveFullModel(_nowralNet, Maps, basePath);
                break;
            case NetworkArchitectureType.LibQuid when _libQuidNet != null:
                LibQuidPersistence.SaveFullModel(_libQuidNet, Maps, basePath);
                break;
            case NetworkArchitectureType.NewNow when _newNowNet != null:
                NewNowPersistence.SaveFullModel(_newNowNet, Maps, basePath);
                break;
            case NetworkArchitectureType.Neuromorphic when _neuroNet != null:
                NeuromorphicModelPersistence.Save(_neuroNet, basePath, _minIn, _maxIn, _minOut, _maxOut);
                break;
            default:
                throw new InvalidOperationException("No hay red activa para guardar.");
        }
        OnLog?.Invoke($"💾 Modelo [{Architecture}] guardado en: {basePath}");
    }

    public void LoadModel(string filePath)
    {
        Stop();
        DisposeNetworks();

        string basePath = filePath.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
            ? filePath[..^5]
            : filePath;

        switch (Architecture)
        {
            case NetworkArchitectureType.Nowral:
                var nowral = NowralPersistence.LoadFullModel(basePath);
                _nowralNet = nowral.Network;
                Maps = nowral.Maps;
                Activation = nowral.Activation;
                _torchOptimizer = CreateTorchOptimizer(_nowralNet.parameters());
                break;

            case NetworkArchitectureType.LibQuid:
                var libQuid = LibQuidPersistence.LoadFullModel(basePath);
                _libQuidNet = libQuid.Network;
                Maps = libQuid.Maps;
                _torchOptimizer = CreateTorchOptimizer(_libQuidNet.parameters());
                break;

            case NetworkArchitectureType.NewNow:
                var newNow = NewNowPersistence.LoadFullModel(basePath);
                _newNowNet = newNow.Network;
                Maps = newNow.Maps;
                Activation = newNow.Activation;
                _torchOptimizer = CreateTorchOptimizer(_newNowNet.parameters());
                break;

            case NetworkArchitectureType.Neuromorphic:
                _neuroNet = NeuromorphicModelPersistence.Load(basePath);
                _minIn = _neuroNet.MinIn;
                _maxIn = _neuroNet.MaxIn;
                _minOut = _neuroNet.MinOut;
                _maxOut = _neuroNet.MaxOut;
                Maps = _neuroNet.Maps.Select(m => new ConnectionMap(m.SourceSize, m.TargetSize)
                {
                    Connections = m.Connections.Select(c => new ConnectionRecord(c.Source, c.Target)).ToList()
                }).ToList();
                break;
        }

        OnTopologyChanged?.Invoke(Maps);
        OnLog?.Invoke($"✅ Modelo [{Architecture}] cargado desde: {Path.GetFileName(filePath)}");
    }

    private void RunTrainingLoop(float targetError, long maxEpochs, CancellationToken token)
    {
        if (Architecture == NetworkArchitectureType.Neuromorphic)
        {
            TrainNeuromorphicLoop(targetError, maxEpochs, token);
            return;
        }

        if (Architecture == NetworkArchitectureType.FastTree || Architecture == NetworkArchitectureType.FastForest)
        {
            if (_treeEnsemble == null || _normInputs == null || _normTargets == null) return;
            if (maxEpochs > 0) _treeEnsemble.NumberOfTrees = (int)Math.Min(maxEpochs, 2000);
            CurrentEpoch = _treeEnsemble.NumberOfTrees;
            LastOpsOrSpikes = _treeEnsemble.NumberOfTrees * Maps[^1].TargetSize;
            CurrentMSE = _treeEnsemble.Train(_normInputs, _normTargets, Settings.LearningRate);
            OnProgressUpdated?.Invoke();
            return;
        }

        if (Architecture == NetworkArchitectureType.LibQuid)
        {
            TrainLibQuidLoop(targetError, maxEpochs, token);
            return;
        }

        if (Architecture == NetworkArchitectureType.Nowral)
        {
            TrainNowralLoop(targetError, maxEpochs, token);
            return;
        }

        if (Architecture == NetworkArchitectureType.Lstm)
        {
            TrainLstmLoop(targetError, maxEpochs, token);
            return;
        }

        if (Architecture == NetworkArchitectureType.StandardMlp || Architecture == NetworkArchitectureType.ResNetTabular)
        {
            TrainStandardTorchLoop(targetError, maxEpochs, token);
            return;
        }

        TrainNewNowLoop(targetError, maxEpochs, token);
    }

    private void TrainStandardTorchLoop(float targetError, long maxEpochs, CancellationToken token)
    {
        if (_torchOptimizer == null || _normInputs == null || _normTargets == null) return;

        int numSamples = _normInputs.Length;
        int batchSize = numSamples > 2000 ? Math.Max(Settings.BatchSize, 512) : Math.Max(1, Settings.BatchSize);
        int inDim = _normInputs[0].Length;
        int outDim = _normTargets[0].Length;
        var rng = new Random(103);
        int[] indices = Enumerable.Range(0, numSamples).ToArray();

        while (!token.IsCancellationRequested)
        {
            float epochError = 0f;
            for (int k = numSamples - 1; k > 0; k--)
            {
                int swapIdx = rng.Next(k + 1);
                (indices[k], indices[swapIdx]) = (indices[swapIdx], indices[k]);
            }

            for (int i = 0; i < numSamples; i += batchSize)
            {
                if (token.IsCancellationRequested) return;
                int currentBatch = Math.Min(batchSize, numSamples - i);
                var bIn = new float[currentBatch * inDim];
                var bOut = new float[currentBatch * outDim];

                for (int b = 0; b < currentBatch; b++)
                {
                    int idx = indices[i + b];
                    Array.Copy(_normInputs[idx], 0, bIn, b * inDim, inDim);
                    Array.Copy(_normTargets[idx], 0, bOut, b * outDim, outDim);
                }

                using var tIn = torch.tensor(bIn, new long[] { currentBatch, inDim }, ScalarType.Float32);
                using var tOut = torch.tensor(bOut, new long[] { currentBatch, outDim }, ScalarType.Float32);
                float loss = Architecture == NetworkArchitectureType.ResNetTabular
                    ? _resNetNet!.TrainStep(tIn, tOut, _torchOptimizer)
                    : _stdMlpNet!.TrainStep(tIn, tOut, _torchOptimizer);
                epochError += loss * currentBatch;
            }

            CurrentEpoch++;
            CurrentMSE = epochError / numSamples;
            OnProgressUpdated?.Invoke();
            if (CurrentMSE <= targetError || (maxEpochs > 0 && CurrentEpoch >= maxEpochs)) break;
        }
    }

    private void TrainLstmLoop(float targetError, long maxEpochs, CancellationToken token)
    {
        if (_lstmNet == null || _torchOptimizer == null || _liquidInputs == null || _liquidTargets == null) return;

        int samples = _liquidInputs.GetLength(0);
        int window = _liquidInputs.GetLength(1);
        int features = _liquidInputs.GetLength(2);
        int outSize = Maps[^1].TargetSize;

        var flatIn = _liquidInputs.Cast<float>().ToArray();
        var flatOut = _liquidTargets.SelectMany(x => x).ToArray();

        using var tIn = torch.tensor(flatIn, new long[] { samples, window, features }).to(ScalarType.Float32);
        using var tOut = torch.tensor(flatOut, new long[] { samples, outSize }).to(ScalarType.Float32);

        while (!token.IsCancellationRequested)
        {
            using var scope = torch.NewDisposeScope();
            CurrentEpoch++;
            CurrentMSE = _lstmNet.TrainStep(tIn, tOut, _torchOptimizer);
            if (CurrentEpoch % 5 == 0) OnProgressUpdated?.Invoke();
            if (CurrentMSE <= targetError || (maxEpochs > 0 && CurrentEpoch >= maxEpochs)) break;
        }
    }

    private void TrainNowralLoop(float targetError, long maxEpochs, CancellationToken token)
    {
        if (_nowralNet == null || _torchOptimizer == null || _normInputs == null || _normTargets == null) return;
        _nowralNet.train();

        var flatIn = _normInputs.SelectMany(x => x).ToArray();
        var flatOut = _normTargets.SelectMany(x => x).ToArray();
        using var tIn = torch.tensor(flatIn, new long[] { _normInputs.Length, _normInputs[0].Length }, ScalarType.Float32);
        using var tOut = torch.tensor(flatOut, new long[] { _normTargets.Length, _normTargets[0].Length }, ScalarType.Float32);

        while (!token.IsCancellationRequested)
        {
            using var scope = torch.NewDisposeScope();
            CurrentEpoch++;
            CurrentMSE = _nowralNet.TrainStep(tIn, tOut, _torchOptimizer);

            if (CurrentEpoch % 10 == 0) OnProgressUpdated?.Invoke();
            if (CurrentMSE <= targetError || (maxEpochs > 0 && CurrentEpoch >= maxEpochs)) break;
        }
    }

    private void TrainLibQuidLoop(float targetError, long maxEpochs, CancellationToken token)
    {
        if (_libQuidNet == null || _torchOptimizer == null || _liquidInputs == null || _liquidTargets == null) return;

        int samples = _liquidInputs.GetLength(0);
        int window = _liquidInputs.GetLength(1);
        int features = _liquidInputs.GetLength(2);
        int outSize = Maps[^1].TargetSize;

        var flatIn = _liquidInputs.Cast<float>().ToArray();
        var flatOut = _liquidTargets.SelectMany(x => x).ToArray();

        using var tIn = torch.tensor(flatIn, new long[] { samples, window, features }).to(ScalarType.Float32);
        using var tOut = torch.tensor(flatOut, new long[] { samples, outSize }).to(ScalarType.Float32);

        while (!token.IsCancellationRequested)
        {
            using var scope = torch.NewDisposeScope();
            CurrentEpoch++;
            CurrentMSE = _libQuidNet.TrainStep(tIn, tOut, _torchOptimizer);

            if (CurrentEpoch % 5 == 0) OnProgressUpdated?.Invoke();
            if (CurrentMSE <= targetError || (maxEpochs > 0 && CurrentEpoch >= maxEpochs)) break;
        }
    }

    private void TrainNewNowLoop(float targetError, long maxEpochs, CancellationToken token)
    {
        if (_newNowNet == null || _torchOptimizer == null || _normInputs == null || _normTargets == null) return;
        _newNowNet.train();

        int numSamples = _normInputs.Length;
        int batchSize = numSamples > 2000 ? Math.Max(Settings.BatchSize, 128) : Math.Max(1, Settings.BatchSize);
        int inDim = _normInputs[0].Length;
        int outDim = _normTargets[0].Length;
        var rng = new Random();
        int[] indices = Enumerable.Range(0, numSamples).ToArray();

        while (!token.IsCancellationRequested)
        {
            float epochError = 0f;
            indices = indices.OrderBy(_ => rng.Next()).ToArray();

            for (int i = 0; i < numSamples; i += batchSize)
            {
                if (token.IsCancellationRequested) return;
                int currentBatch = Math.Min(batchSize, numSamples - i);
                var bIn = new float[currentBatch * inDim];
                var bOut = new float[currentBatch * outDim];

                for (int b = 0; b < currentBatch; b++)
                {
                    int idx = indices[i + b];
                    Array.Copy(_normInputs[idx], 0, bIn, b * inDim, inDim);
                    Array.Copy(_normTargets[idx], 0, bOut, b * outDim, outDim);
                }

                using var tIn = torch.tensor(bIn, new long[] { currentBatch, inDim }, ScalarType.Float32);
                using var tOut = torch.tensor(bOut, new long[] { currentBatch, outDim }, ScalarType.Float32);
                float loss = _newNowNet.TrainStep(tIn, tOut, _torchOptimizer);
                epochError += loss * currentBatch;
            }

            CurrentEpoch++;
            CurrentMSE = epochError / numSamples;
            OnProgressUpdated?.Invoke();
            if (CurrentEpoch % 10 == 0) OnLog?.Invoke($"Época {CurrentEpoch} - MSE: {CurrentMSE:F8}");
            if (CurrentMSE <= targetError || (maxEpochs > 0 && CurrentEpoch >= maxEpochs)) break;
        }
    }

    private void TrainNeuromorphicLoop(float targetError, long maxEpochs, CancellationToken token)
    {
        if (_neuroNet == null || _normInputs == null || _normTargets == null) return;

        int numSamples = _normInputs.Length;
        int batchSize = numSamples > 2000 ? Math.Max(Settings.BatchSize, 256) : Math.Max(1, Settings.BatchSize);
        float lr = (float)Settings.LearningRate * (batchSize >= 256 ? 1.5f : 1.0f);
        float stdp = (float)Settings.StdpRate;
        var rng = new Random(103);
        int[] indices = Enumerable.Range(0, numSamples).ToArray();
        var bInFull = new float[batchSize][];
        var bOutFull = new float[batchSize][];

        while (!token.IsCancellationRequested)
        {
            float epochError = 0f;
            // Fisher-Yates in-place shuffle (cero asignaciones)
            for (int k = numSamples - 1; k > 0; k--)
            {
                int swapIdx = rng.Next(k + 1);
                (indices[k], indices[swapIdx]) = (indices[swapIdx], indices[k]);
            }

            for (int i = 0; i < numSamples; i += batchSize)
            {
                if (token.IsCancellationRequested) return;
                int currentBatch = Math.Min(batchSize, numSamples - i);
                var bIn = currentBatch == batchSize ? bInFull : new float[currentBatch][];
                var bOut = currentBatch == batchSize ? bOutFull : new float[currentBatch][];

                for (int b = 0; b < currentBatch; b++)
                {
                    int idx = indices[i + b];
                    bIn[b] = _normInputs[idx];
                    bOut[b] = _normTargets[idx];
                }

                float loss = _neuroNet.TrainStep(bIn, bOut, lr, stdp);
                epochError += loss * currentBatch;
            }

            CurrentEpoch++;
            CurrentMSE = epochError / numSamples;
            LastOpsOrSpikes = _neuroNet.LastSpikesProcessed;
            OnProgressUpdated?.Invoke();

            if (CurrentEpoch % 10 == 0)
                OnLog?.Invoke($"[SNN] Época {CurrentEpoch} - MSE: {CurrentMSE:F8} | Espigas/muestra: {LastOpsOrSpikes}");
            if (CurrentMSE <= targetError || (maxEpochs > 0 && CurrentEpoch >= maxEpochs)) break;
        }
    }

    public static (float[][] Inputs, float[][] Targets) GenerateSyntheticDataset(int inputDim, int outputDim, int sampleCount = 240)
    {
        var rng = new Random(103);
        var inputs = new float[sampleCount][];
        var targets = new float[sampleCount][];

        for (int i = 0; i < sampleCount; i++)
        {
            inputs[i] = new float[inputDim];
            targets[i] = new float[outputDim];
            float sum = 0f;
            for (int d = 0; d < inputDim; d++)
            {
                float v = (float)rng.NextDouble();
                inputs[i][d] = v;
                sum += MathF.Sin(v * MathF.PI * (d + 1));
            }
            for (int o = 0; o < outputDim; o++)
            {
                targets[i][o] = Math.Clamp(0.5f + 0.35f * (sum / Math.Max(1, inputDim)) + 0.05f * o, 0.05f, 0.95f);
            }
        }
        return (inputs, targets);
    }

    private void PrepareLiquidSequences()
    {
        if (_normInputs == null || _normTargets == null || Maps.Count == 0) return;
        int inSize = Maps[0].SourceSize;
        int count = _normInputs.Length - LiquidWindowSize + 1;
        if (count <= 0)
        {
            _liquidInputs = null;
            _liquidTargets = null;
            OnLog?.Invoke("⚠️ Datos insuficientes para ventana temporal LibQuid (mínimo 5 filas).");
            return;
        }

        _liquidInputs = new float[count, LiquidWindowSize, inSize];
        _liquidTargets = new float[count][];
        for (int i = 0; i < count; i++)
        {
            for (int t = 0; t < LiquidWindowSize; t++)
                for (int f = 0; f < inSize; f++)
                    _liquidInputs[i, t, f] = _normInputs[i + t][f];
            _liquidTargets[i] = _normTargets[i + LiquidWindowSize - 1];
        }
    }

    private void LogNormalizationReport(int inputCount, int outputCount)
    {
        if (_minIn == null || _maxIn == null || _minOut == null || _maxOut == null) return;
        OnLog?.Invoke("--- REPORTE DE NORMALIZACIÓN (MIN / MAX) ---");
        for (int i = 0; i < inputCount; i++)
            OnLog?.Invoke($"Input Col {i}: [{_minIn[i]:F2} .. {_maxIn[i]:F2}] -> [0 .. 1]");
        for (int i = 0; i < outputCount; i++)
            OnLog?.Invoke($"Target Col {i}: [{_minOut[i]:F2} .. {_maxOut[i]:F2}] -> [0 .. 1]");
    }

    private torch.optim.Optimizer CreateTorchOptimizer(IEnumerable<TorchSharp.Modules.Parameter> parameters)
    {
        return Settings.Optimizer switch
        {
            OptimizerType.Adam => torch.optim.Adam(parameters, Settings.LearningRate),
            OptimizerType.SGD => torch.optim.SGD(parameters, Settings.LearningRate),
            OptimizerType.RMSprop => torch.optim.RMSProp(parameters, Settings.LearningRate),
            _ => torch.optim.Adam(parameters, Settings.LearningRate)
        };
    }

    private static float[][] Normalize(float[][] data, out float[] min, out float[] max)
    {
        int rows = data.Length;
        int cols = data[0].Length;
        min = new float[cols];
        max = new float[cols];
        var normalized = new float[rows][];

        for (int j = 0; j < cols; j++)
        {
            min[j] = data.Select(r => r[j]).Min();
            max[j] = data.Select(r => r[j]).Max();
        }

        for (int i = 0; i < rows; i++)
        {
            normalized[i] = new float[cols];
            for (int j = 0; j < cols; j++)
            {
                float range = max[j] - min[j];
                normalized[i][j] = range == 0 ? 0f : (data[i][j] - min[j]) / range;
            }
        }
        return normalized;
    }

    private static ConnectionMap CloneMap(ConnectionMap map) => new(map.SourceSize, map.TargetSize)
    {
        Connections = map.Connections.Select(c => new ConnectionRecord(c.Source, c.Target)).ToList()
    };

    private static List<NeuromorphicLayerMap> ToNeuromorphicMaps(List<ConnectionMap> maps)
    {
        return maps.Select(m => new NeuromorphicLayerMap(
            m.SourceSize,
            m.TargetSize,
            m.Connections.Select(c => (c.Source, c.Target)))).ToList();
    }

    private void DisposeNetworks()
    {
        _torchOptimizer?.Dispose();
        _torchOptimizer = null;
        _nowralNet?.Dispose();
        _nowralNet = null;
        _libQuidNet?.Dispose();
        _libQuidNet = null;
        _newNowNet?.Dispose();
        _newNowNet = null;
        _neuroNet = null;
        _treeEnsemble?.Dispose();
        _treeEnsemble = null;
        _stdMlpNet?.Dispose();
        _stdMlpNet = null;
        _resNetNet?.Dispose();
        _resNetNet = null;
        _lstmNet?.Dispose();
        _lstmNet = null;
    }

    public void Dispose()
    {
        Stop();
        DisposeNetworks();
    }
}

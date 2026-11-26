using System;
using System.Collections.Generic;
using System.Threading;
using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Training;

/// <summary>
/// Red Neuronal Neuromórfica Entrenable (SNN) basada en neuronas LIF, codificación TTFS,
/// propagación orientada a eventos (espigas) y aprendizaje híbrido Surrogate-Gradient + R-STDP.
/// Optimizada para CPU con arreglos planos 1D, SIMD (Vector&lt;float&gt;), ordenamiento inline y paralelismo por lotes.
/// </summary>
public sealed partial class NeuromorphicTrainableNetwork : IDisposable
{
    private readonly List<SpikingLayerState> _layers = new();
    private readonly CognitiveConfig _config;
    private readonly ThreadLocal<ThreadScratch> _scratchLocal;
    private readonly float[][] _totalGradW;
    private readonly float[][] _totalGradB;
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.NoRecursion);
    private readonly object _lifecycleLock = new();
    private int _activeOperations;
    private bool _disposed;
    private int _adamStep;
    internal int AdamStep { get => _adamStep; set => _adamStep = value; }
    private long _lastSpikesProcessed;
    private long _lastLifFiredCount;
    private long _lastPropagatedEventCount;
    private long _lastSynapticOpsCount;

    public IReadOnlyList<SpikingLayerState> Layers => _layers;
    public IReadOnlyList<NeuromorphicLayerMap> Maps { get; }
    public CognitiveConfig Config => _config;
    public NeuromorphicExecutionMode ExecutionMode => _config.ExecutionMode;

    /// <summary>
    /// Último conteo global de espigas registrado por el motor (espigas TTFS de entrada + disparos LIF ocultos).
    /// En llamadas concurrentes a <see cref="Predict"/>, refleja de forma atómica la última operación completada.
    /// </summary>
    public long LastSpikesProcessed => Interlocked.Read(ref _lastSpikesProcessed);

    /// <summary>
    /// Métrica global de la última operación completada: número de neuronas ocultas LIF que superaron su umbral y dispararon.
    /// </summary>
    public long LastLifFiredCount => Interlocked.Read(ref _lastLifFiredCount);

    /// <summary>
    /// Métrica global de la última operación completada: total de eventos propagados entre capas (entrada TTFS + eventos enviados a capas siguientes).
    /// </summary>
    public long LastPropagatedEventCount => Interlocked.Read(ref _lastPropagatedEventCount);

    /// <summary>
    /// Métrica global de la última operación completada: operaciones sinápticas recorridas por evento mediante <c>ConnectedTargets</c> (cuenta operaciones por evento, no conexiones únicas).
    /// </summary>
    public long LastSynapticOpsCount => Interlocked.Read(ref _lastSynapticOpsCount);
    public float[]? MinIn { get; set; }
    public float[]? MaxIn { get; set; }
    public float[]? MinOut { get; set; }
    public float[]? MaxOut { get; set; }

    public NeuromorphicTrainableNetwork(
        IReadOnlyList<NeuromorphicLayerMap> maps,
        CognitiveConfig? config = null,
        int seed = 103)
    {
        ArgumentNullException.ThrowIfNull(maps);
        if (maps.Count == 0)
            throw new ArgumentException("La topología debe contener al menos una capa.", nameof(maps));

        for (int i = 0; i < maps.Count; i++)
        {
            if (maps[i] == null)
                throw new ArgumentNullException(nameof(maps), $"La capa en el índice {i} no puede ser nula.");
            if (maps[i].SourceSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(maps), maps[i].SourceSize, $"SourceSize de la capa {i} debe ser mayor que 0.");
            if (maps[i].TargetSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(maps), maps[i].TargetSize, $"TargetSize de la capa {i} debe ser mayor que 0.");
            if (i > 0 && maps[i - 1].TargetSize != maps[i].SourceSize)
                throw new ArgumentException($"Incompatibilidad de dimensiones entre capa {i - 1} (TargetSize={maps[i - 1].TargetSize}) y capa {i} (SourceSize={maps[i].SourceSize}).", nameof(maps));
        }

        Maps = maps;
        _config = config ?? new CognitiveConfig
        {
            BaseThreshold = 0.25f,
            MembraneDecay = 0.90f,
            LateralInhibitionFactor = 0.05f,
            ThresholdAdaptationRate = 0.02f,
            TimeWindow = 1.0f
        };
        ValidateConfig(_config);

        var rng = new Random(seed);
        foreach (var map in maps)
        {
            _layers.Add(new SpikingLayerState(map, _config, rng));
        }

        _totalGradW = new float[_layers.Count][];
        _totalGradB = new float[_layers.Count][];
        for (int l = 0; l < _layers.Count; l++)
        {
            _totalGradW[l] = new float[_layers[l].OutputDim * _layers[l].InputDim];
            _totalGradB[l] = new float[_layers[l].OutputDim];
        }

        _scratchLocal = new ThreadLocal<ThreadScratch>(() => new ThreadScratch(_layers), trackAllValues: true);
    }

    private static void ValidateConfig(CognitiveConfig cfg)
    {
        if (!float.IsFinite(cfg.BaseThreshold) || cfg.BaseThreshold <= 0f)
            throw new ArgumentOutOfRangeException(nameof(cfg.BaseThreshold), cfg.BaseThreshold, "BaseThreshold debe ser finito y mayor que 0.");
        if (!float.IsFinite(cfg.MembraneDecay) || cfg.MembraneDecay < 0f || cfg.MembraneDecay > 1f)
            throw new ArgumentOutOfRangeException(nameof(cfg.MembraneDecay), cfg.MembraneDecay, "MembraneDecay debe estar en [0, 1].");
        if (!float.IsFinite(cfg.LateralInhibitionFactor) || cfg.LateralInhibitionFactor < 0f)
            throw new ArgumentOutOfRangeException(nameof(cfg.LateralInhibitionFactor), cfg.LateralInhibitionFactor, "LateralInhibitionFactor debe ser finito y >= 0.");
        if (!float.IsFinite(cfg.ThresholdAdaptationRate) || cfg.ThresholdAdaptationRate < 0f)
            throw new ArgumentOutOfRangeException(nameof(cfg.ThresholdAdaptationRate), cfg.ThresholdAdaptationRate, "ThresholdAdaptationRate debe ser finito y >= 0.");
        if (!float.IsFinite(cfg.ThresholdDecay) || cfg.ThresholdDecay < 0f || cfg.ThresholdDecay > 1f)
            throw new ArgumentOutOfRangeException(nameof(cfg.ThresholdDecay), cfg.ThresholdDecay, "ThresholdDecay debe estar en [0, 1].");
        if (cfg.MaxDeliberationSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(cfg.MaxDeliberationSteps), cfg.MaxDeliberationSteps, "MaxDeliberationSteps debe ser mayor que 0.");
        if (!float.IsFinite(cfg.EarlyExitConfidenceThreshold) || cfg.EarlyExitConfidenceThreshold < 0f)
            throw new ArgumentOutOfRangeException(nameof(cfg.EarlyExitConfidenceThreshold), cfg.EarlyExitConfidenceThreshold, "EarlyExitConfidenceThreshold debe ser finito y >= 0.");
        if (!float.IsFinite(cfg.TimeWindow) || cfg.TimeWindow <= 0f)
            throw new ArgumentOutOfRangeException(nameof(cfg.TimeWindow), cfg.TimeWindow, "TimeWindow debe ser finito y mayor que 0.");
        if (!Enum.IsDefined(cfg.ExecutionMode))
            throw new ArgumentOutOfRangeException(nameof(cfg.ExecutionMode), cfg.ExecutionMode, "ExecutionMode inválido.");
        if (!Enum.IsDefined(cfg.EncodingMode))
            throw new ArgumentOutOfRangeException(nameof(cfg.EncodingMode), cfg.EncodingMode, "EncodingMode inválido.");
        if (!float.IsFinite(cfg.GlobalEncodingScale) || cfg.GlobalEncodingScale <= 0f)
            throw new ArgumentOutOfRangeException(nameof(cfg.GlobalEncodingScale), cfg.GlobalEncodingScale, "GlobalEncodingScale debe ser finito y mayor que 0.");
    }

    private void ThrowIfDisposed()
    {
        lock (_lifecycleLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }

    private void BeginOperation()
    {
        lock (_lifecycleLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _activeOperations++;
        }
    }

    private void EndOperation()
    {
        lock (_lifecycleLock)
        {
            _activeOperations--;
            if (_disposed && _activeOperations == 0)
                Monitor.PulseAll(_lifecycleLock);
        }
    }

    private void EnterReadLockSafe()
    {
        BeginOperation();
        try
        {
            _rwLock.EnterReadLock();
        }
        catch
        {
            EndOperation();
            throw;
        }
    }

    private void ExitReadLockSafe()
    {
        try
        {
            _rwLock.ExitReadLock();
        }
        finally
        {
            EndOperation();
        }
    }

    private void EnterWriteLockSafe()
    {
        BeginOperation();
        try
        {
            _rwLock.EnterWriteLock();
        }
        catch
        {
            EndOperation();
            throw;
        }
    }

    private void ExitWriteLockSafe()
    {
        try
        {
            _rwLock.ExitWriteLock();
        }
        finally
        {
            EndOperation();
        }
    }

    public void Dispose()
    {
        lock (_lifecycleLock)
        {
            if (_disposed)
                return;

            _disposed = true;
            while (_activeOperations > 0)
                Monitor.Wait(_lifecycleLock);
        }

        _scratchLocal.Dispose();
        _rwLock.Dispose();
    }

    internal void ExecuteWithReadLock(Action action)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(action);
        EnterReadLockSafe();
        try
        {
            action();
        }
        finally
        {
            ExitReadLockSafe();
        }
    }

    private static void ValidateFiniteValues(ReadOnlySpan<float> values, string paramName)
    {
        for (int i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
                throw new ArgumentOutOfRangeException(paramName, values[i], $"El elemento en el índice {i} contiene NaN o Infinito.");
        }
    }
}

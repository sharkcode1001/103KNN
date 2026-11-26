using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Neurons;

/// <summary>
/// Neurona Leaky Integrate-and-Fire adaptada para procesamiento cognitivo y razonamiento secuencial.
/// Incluye adaptación de umbral homeostática, inhibición lateral y registro de primer tiempo de disparo (TTFS).
/// </summary>
public sealed class CognitiveLifNeuron
{
    private readonly int _neuronId;
    private readonly CognitiveConfig _config;

    private float _voltage;
    private float _adaptiveThreshold;
    private float _firstSpikeTime;
    private float _lastTimestamp;
    private int _spikesFiredCount;
    private bool _hasFiredInEpisode;

    public CognitiveLifNeuron(int neuronId, CognitiveConfig? config = null)
    {
        _neuronId = neuronId;
        _config = config ?? new CognitiveConfig();
        _adaptiveThreshold = _config.BaseThreshold;
        _firstSpikeTime = float.PositiveInfinity;
        _lastTimestamp = 0.0f;
    }

    public int NeuronId => _neuronId;
    public float Voltage => _voltage;
    public float Threshold => _adaptiveThreshold;
    public float FirstSpikeTime => _firstSpikeTime;
    public float LastTimestamp => _lastTimestamp;
    public int SpikesFiredCount => _spikesFiredCount;
    public bool HasFiredInEpisode => _hasFiredInEpisode;

    private static float ComputeTimeDecay(float dt, float decay)
    {
        if (dt <= 0.0f || decay >= 1.0f) return 1.0f;
        if (decay <= 0.0f) return 0.0f;
        return MathF.Exp(dt * MathF.Log(decay));
    }

    /// <summary>
    /// Integra una corriente sináptica entrante en el instante especificado usando decaimiento exponencial continuo ($\lambda^{\Delta t}$).
    /// Para $\Delta t = 0$, el factor de decaimiento es exactamente $1.0$.
    /// </summary>
    public bool Integrate(float synapticCurrent, float timestamp)
    {
        if (!float.IsFinite(synapticCurrent))
            throw new ArgumentOutOfRangeException(nameof(synapticCurrent), synapticCurrent, "La corriente sináptica debe ser finita.");
        if (!float.IsFinite(timestamp))
            throw new ArgumentOutOfRangeException(nameof(timestamp), timestamp, "El timestamp debe ser finito.");

        float dt = MathF.Max(0.0f, timestamp - _lastTimestamp);
        if (dt > 0.0f)
        {
            _voltage *= ComputeTimeDecay(dt, _config.MembraneDecay);
            _adaptiveThreshold = _config.BaseThreshold +
                                 (_adaptiveThreshold - _config.BaseThreshold) * ComputeTimeDecay(dt, _config.ThresholdDecay);
            _lastTimestamp = timestamp;
        }

        _voltage += synapticCurrent;

        if (_voltage >= _adaptiveThreshold)
        {
            _spikesFiredCount++;

            if (!_hasFiredInEpisode)
            {
                _hasFiredInEpisode = true;
                _firstSpikeTime = timestamp;
            }

            _voltage = 0.0f;
            _adaptiveThreshold += _config.ThresholdAdaptationRate;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Evalúa el cierre de la ventana temporal aplicando el decaimiento restante hasta <paramref name="timeWindow"/> y el sesgo.
    /// </summary>
    public bool IntegrateEndOfWindow(float timeWindow, float bias = 0.0f)
    {
        if (!float.IsFinite(timeWindow))
            throw new ArgumentOutOfRangeException(nameof(timeWindow), timeWindow, "TimeWindow debe ser finito.");
        if (!float.IsFinite(bias))
            throw new ArgumentOutOfRangeException(nameof(bias), bias, "Bias debe ser finito.");

        float dtEnd = MathF.Max(0.0f, timeWindow - _lastTimestamp);
        float endLeak = ComputeTimeDecay(dtEnd, _config.MembraneDecay);
        float v = (_voltage * endLeak) + bias;
        _lastTimestamp = MathF.Max(_lastTimestamp, timeWindow);

        if (v >= _adaptiveThreshold && !_hasFiredInEpisode)
        {
            _spikesFiredCount++;
            _hasFiredInEpisode = true;
            _firstSpikeTime = timeWindow;
            _voltage = 0.0f;
            return true;
        }

        _voltage = v;
        return false;
    }

    /// <summary>
    /// Aplica inhibición lateral cuando otra neurona competidora dispara.
    /// </summary>
    public void ApplyLateralInhibition(float factor = 1.0f)
    {
        // Se reduce el voltaje y se incrementa transitoriamente el umbral
        _voltage *= MathF.Max(0.0f, 1.0f - (_config.LateralInhibitionFactor * factor));
        _adaptiveThreshold += _config.LateralInhibitionFactor * 0.5f * factor;
    }

    /// <summary>
    /// Avanza un ciclo de decaimiento temporal pasivo.
    /// </summary>
    public void Step()
    {
        _voltage *= _config.MembraneDecay;

        // Recuperación gradual del umbral hacia el valor base
        if (_adaptiveThreshold > _config.BaseThreshold)
        {
            _adaptiveThreshold = (_adaptiveThreshold * _config.ThresholdDecay) +
                                 (_config.BaseThreshold * (1.0f - _config.ThresholdDecay));
        }
    }

    /// <summary>
    /// Reinicia el estado de la neurona para una nueva evaluación o episodio cognitivo.
    /// </summary>
    public void Reset()
    {
        _voltage = 0.0f;
        _adaptiveThreshold = _config.BaseThreshold;
        _firstSpikeTime = float.PositiveInfinity;
        _lastTimestamp = 0.0f;
        _spikesFiredCount = 0;
        _hasFiredInEpisode = false;
    }
}

using System.Diagnostics;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Encoding;
using _103KNN.Neuromorphic.Neurons;

namespace _103KNN.Neuromorphic.Associative;

public sealed record NeuromorphicKnnResult(
    int PredictedClass,
    IReadOnlyList<(KnnSample Sample, float SpikeTime)> NearestNeighbors,
    int TotalSpikesProcessed,
    TimeSpan ElapsedTime
);

public sealed record KnnBenchmarkResult(
    int NeuromorphicPrediction,
    int ClassicPrediction,
    bool PredictionsMatch,
    long ClassicFlops,
    long NeuromorphicSops,
    TimeSpan ClassicTime,
    TimeSpan NeuromorphicTime
);

/// <summary>
/// Motor k-NN Neuromórfico basado en memoria asociativa y Time-to-First-Spike.
/// Encuentra los K vecinos más cercanos midiendo el orden de disparo temporal en lugar de calcular
/// distancias euclidianas densas con todos los puntos.
/// </summary>
public sealed class NeuromorphicKnnEngine
{
    private readonly int _featureDim;
    private readonly List<KnnSample> _samples = new();
    private readonly List<CognitiveLifNeuron> _neurons = new();
    private readonly ISpikeEncoder _encoder;
    private readonly CognitiveConfig _config;

    public NeuromorphicKnnEngine(int featureDim, CognitiveConfig? config = null)
    {
        _featureDim = featureDim;
        _config = config ?? new CognitiveConfig
        {
            BaseThreshold = 0.85f,
            MembraneDecay = 0.92f,
            LateralInhibitionFactor = 0.05f
        };
        _encoder = new TimeToFirstSpikeEncoder();
    }

    public int FeatureDim => _featureDim;
    public int SampleCount => _samples.Count;
    public IReadOnlyList<KnnSample> Samples => _samples;

    public void AddSample(KnnSample sample)
    {
        if (sample.Features.Length != _featureDim)
            throw new ArgumentException($"La dimensión de la muestra ({sample.Features.Length}) no coincide con la esperada ({_featureDim}).");

        _samples.Add(sample);
        _neurons.Add(new CognitiveLifNeuron(_neurons.Count, _config));
    }

    public void Clear()
    {
        _samples.Clear();
        _neurons.Clear();
    }

    /// <summary>
    /// Clasifica una consulta encontrando los K vecinos más tempranos en disparar.
    /// </summary>
    public NeuromorphicKnnResult Classify(ReadOnlySpan<float> query, int k = 3)
    {
        if (_samples.Count == 0)
            throw new InvalidOperationException("No hay muestras cargadas en el motor k-NN.");

        k = Math.Clamp(k, 1, _samples.Count);
        var sw = Stopwatch.StartNew();

        // 1. Reiniciar estado de membrana
        for (int i = 0; i < _neurons.Count; i++)
        {
            _neurons[i].Reset();
        }

        // 2. Codificar la consulta en tren de espigas ordenado temporalmente (TTFS)
        var querySpikes = _encoder.Encode(query, _config.TimeWindow);
        int totalSpikes = querySpikes.Count;

        // 3. Propagación por eventos: solo las características activas propagan señal
        var firedUnits = new List<(KnnSample Sample, float SpikeTime)>();

        foreach (var spike in querySpikes)
        {
            int featIdx = spike.FeatureIndex;
            float mag = spike.Magnitude * spike.Polarity;

            for (int i = 0; i < _samples.Count; i++)
            {
                // Peso sináptico proporcional a la característica memorizada
                float weight = _samples[i].Features[featIdx];
                float current = weight * mag;

                if (_neurons[i].Integrate(current, spike.Timestamp))
                {
                    firedUnits.Add((_samples[i], spike.Timestamp));
                }
            }
        }

        // Si menos de K dispararon formalmente, completar con las de mayor potencial acumulado
        if (firedUnits.Count < k)
        {
            var remaining = _samples
                .Select((s, idx) => (Sample: s, Time: _neurons[idx].HasFiredInEpisode ? _neurons[idx].FirstSpikeTime : (10.0f - _neurons[idx].Voltage)))
                .Where(item => !firedUnits.Any(f => f.Sample.Id == item.Sample.Id))
                .OrderBy(item => item.Time)
                .Take(k - firedUnits.Count);

            firedUnits.AddRange(remaining);
        }

        var nearest = firedUnits.OrderBy(x => x.SpikeTime).Take(k).ToList();

        // Votación mayoritaria entre los K primeros en disparar
        int predictedClass = nearest
            .GroupBy(x => x.Sample.ClassLabel)
            .OrderByDescending(g => g.Count())
            .First()
            .Key;

        sw.Stop();

        return new NeuromorphicKnnResult(
            PredictedClass: predictedClass,
            NearestNeighbors: nearest,
            TotalSpikesProcessed: totalSpikes,
            ElapsedTime: sw.Elapsed
        );
    }

    /// <summary>
    /// Compara directamente el rendimiento y conteo de operaciones contra k-NN Euclidiano clásico.
    /// </summary>
    public KnnBenchmarkResult BenchmarkVsClassic(ReadOnlySpan<float> query, int k = 3)
    {
        // 1. Ejecutar k-NN Euclidiano Clásico
        var swClassic = Stopwatch.StartNew();
        long classicFlops = 0;
        var distances = new List<(KnnSample Sample, float Distance)>(_samples.Count);

        for (int i = 0; i < _samples.Count; i++)
        {
            float sumSq = 0.0f;
            for (int d = 0; d < _featureDim; d++)
            {
                float diff = query[d] - _samples[i].Features[d];
                sumSq += diff * diff;
                classicFlops += 3; // resta, multiplicación, suma
            }
            classicFlops += 1; // raíz cuadrada (aprox)
            distances.Add((_samples[i], MathF.Sqrt(sumSq)));
        }

        int classicWinner = distances
            .OrderBy(x => x.Distance)
            .Take(k)
            .GroupBy(x => x.Sample.ClassLabel)
            .OrderByDescending(g => g.Count())
            .First()
            .Key;
        swClassic.Stop();

        // 2. Ejecutar k-NN Neuromórfico
        var neuroRes = Classify(query, k);
        // En neuromórfico, cada espiga activa realiza solo acumulaciones (sumas) en lugar de restas+cuadrados+raíz
        long neuroSops = (long)neuroRes.TotalSpikesProcessed * _samples.Count;

        return new KnnBenchmarkResult(
            NeuromorphicPrediction: neuroRes.PredictedClass,
            ClassicPrediction: classicWinner,
            PredictionsMatch: neuroRes.PredictedClass == classicWinner,
            ClassicFlops: classicFlops,
            NeuromorphicSops: neuroSops,
            ClassicTime: swClassic.Elapsed,
            NeuromorphicTime: neuroRes.ElapsedTime
        );
    }
}

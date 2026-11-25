using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Encoding;
using _103KNN.Neuromorphic.Neurons;

namespace _103KNN.Neuromorphic.Training;

/// <summary>
/// Especificación de topología entre dos capas para el motor neuromórfico entrenable.
/// </summary>
public sealed class NeuromorphicLayerMap
{
    public int SourceSize { get; set; }
    public int TargetSize { get; set; }
    public List<(int Source, int Target)> Connections { get; set; } = new();

    public NeuromorphicLayerMap() { }

    public NeuromorphicLayerMap(int sourceSize, int targetSize, IEnumerable<(int Source, int Target)>? connections = null)
    {
        SourceSize = sourceSize;
        TargetSize = targetSize;
        if (connections != null)
        {
            Connections.AddRange(connections);
        }
        else
        {
            for (int s = 0; s < sourceSize; s++)
                for (int t = 0; t < targetSize; t++)
                    Connections.Add((s, t));
        }
    }
}

/// <summary>
/// Red Neuronal Neuromórfica Entrenable (SNN) basada en neuronas LIF, codificación TTFS,
/// propagación orientada a eventos (espigas) y aprendizaje híbrido Surrogate-Gradient + R-STDP.
/// Optimizada para CPU mediante adyacencia dispersa y paralelización por mini-lotes.
/// </summary>
public sealed class NeuromorphicTrainableNetwork
{
    public sealed class SpikingLayerState
    {
        public int InputDim { get; }
        public int OutputDim { get; }
        public bool[,] Mask { get; }
        public int[][] ConnectedTargets { get; }
        public float[,] Weights { get; }
        public float[] Biases { get; }
        public float[,] AdamM { get; }
        public float[,] AdamV { get; }
        public float[] BiasM { get; }
        public float[] BiasV { get; }
        public CognitiveLifNeuron[] Neurons { get; }

        public SpikingLayerState(NeuromorphicLayerMap map, CognitiveConfig config, Random rng)
        {
            InputDim = map.SourceSize;
            OutputDim = map.TargetSize;
            Mask = new bool[OutputDim, InputDim];
            Weights = new float[OutputDim, InputDim];
            Biases = new float[OutputDim];
            AdamM = new float[OutputDim, InputDim];
            AdamV = new float[OutputDim, InputDim];
            BiasM = new float[OutputDim];
            BiasV = new float[OutputDim];
            Neurons = new CognitiveLifNeuron[OutputDim];

            var perSource = new List<int>[InputDim];
            for (int i = 0; i < InputDim; i++) perSource[i] = new List<int>();

            foreach (var (src, dst) in map.Connections)
            {
                if (src >= 0 && src < InputDim && dst >= 0 && dst < OutputDim && !Mask[dst, src])
                {
                    Mask[dst, src] = true;
                    perSource[src].Add(dst);
                }
            }

            ConnectedTargets = new int[InputDim][];
            for (int i = 0; i < InputDim; i++)
                ConnectedTargets[i] = perSource[i].ToArray();

            float stdDev = MathF.Sqrt(2.0f / Math.Max(1, InputDim + OutputDim)) * 1.4f;
            for (int j = 0; j < OutputDim; j++)
            {
                Neurons[j] = new CognitiveLifNeuron(j, config);
                for (int i = 0; i < InputDim; i++)
                {
                    if (Mask[j, i])
                    {
                        Weights[j, i] = (float)((rng.NextDouble() * 2.0 - 1.0) * stdDev);
                    }
                }
            }
        }
    }

    private sealed class ThreadScratch
    {
        public readonly float[][,] GradW;
        public readonly float[][] GradB;
        public readonly float[][] LayerActs;
        public readonly float[][] LayerNets;
        public readonly float[][] LayerSpikeTimes;
        public readonly float[][] NeuronVoltage;
        public readonly float[][] NeuronThreshold;
        public readonly bool[][] NeuronFired;
        public readonly Spike[][] SpikeBuffers;
        public float MseSum;
        public long SpikesSum;

        public ThreadScratch(IReadOnlyList<SpikingLayerState> layers)
        {
            int count = layers.Count;
            GradW = new float[count][,];
            GradB = new float[count][];
            LayerActs = new float[count + 1][];
            LayerNets = new float[count][];
            LayerSpikeTimes = new float[count + 1][];
            NeuronVoltage = new float[count][];
            NeuronThreshold = new float[count][];
            NeuronFired = new bool[count][];
            SpikeBuffers = new Spike[count + 1][];

            LayerActs[0] = new float[layers[0].InputDim];
            LayerSpikeTimes[0] = new float[layers[0].InputDim];
            SpikeBuffers[0] = new Spike[layers[0].InputDim];

            for (int l = 0; l < count; l++)
            {
                int inD = layers[l].InputDim;
                int outD = layers[l].OutputDim;
                GradW[l] = new float[outD, inD];
                GradB[l] = new float[outD];
                LayerActs[l + 1] = new float[outD];
                LayerNets[l] = new float[outD];
                LayerSpikeTimes[l + 1] = new float[outD];
                NeuronVoltage[l] = new float[outD];
                NeuronThreshold[l] = new float[outD];
                NeuronFired[l] = new bool[outD];
                SpikeBuffers[l + 1] = new Spike[outD];
            }
        }

        public void ClearGrads()
        {
            MseSum = 0f;
            SpikesSum = 0;
            for (int l = 0; l < GradW.Length; l++)
            {
                Array.Clear(GradW[l]);
                Array.Clear(GradB[l]);
            }
        }
    }

    private readonly List<SpikingLayerState> _layers = new();
    private readonly CognitiveConfig _config;
    private readonly ThreadLocal<ThreadScratch> _scratchLocal;
    private int _adamStep;

    public IReadOnlyList<SpikingLayerState> Layers => _layers;
    public IReadOnlyList<NeuromorphicLayerMap> Maps { get; }
    public CognitiveConfig Config => _config;
    public long LastSpikesProcessed { get; private set; }

    public NeuromorphicTrainableNetwork(
        IReadOnlyList<NeuromorphicLayerMap> maps,
        CognitiveConfig? config = null,
        int seed = 103)
    {
        if (maps == null || maps.Count == 0)
            throw new ArgumentException("La topología debe contener al menos una capa.", nameof(maps));

        Maps = maps;
        _config = config ?? new CognitiveConfig
        {
            BaseThreshold = 0.25f,
            MembraneDecay = 0.90f,
            LateralInhibitionFactor = 0.05f,
            ThresholdAdaptationRate = 0.02f,
            TimeWindow = 1.0f
        };

        var rng = new Random(seed);
        foreach (var map in maps)
        {
            _layers.Add(new SpikingLayerState(map, _config, rng));
        }

        _scratchLocal = new ThreadLocal<ThreadScratch>(() => new ThreadScratch(_layers), trackAllValues: true);
    }

    /// <summary>
    /// Ejecuta una inferencia neuromórfica por eventos (espigas TTFS + integración LIF)
    /// y devuelve los valores predichos en [0, 1] junto con el conteo de espigas procesadas.
    /// </summary>
    public float[] Predict(ReadOnlySpan<float> input)
    {
        var scratch = _scratchLocal.Value!;
        long spikes = ForwardFast(input, scratch);
        LastSpikesProcessed = spikes;
        return (float[])scratch.LayerActs[_layers.Count].Clone();
    }

    /// <summary>
    /// Ejecuta un paso de entrenamiento sobre un mini-lote usando Surrogate-Gradient + modulación STDP
    /// paralelizado en CPU entre las muestras del lote.
    /// </summary>
    public float TrainStep(float[][] batchInputs, float[][] batchTargets, float learningRate = 0.005f, float stdpRate = 0.002f)
    {
        int batchSize = batchInputs.Length;
        if (batchSize == 0) return 0f;

        int layerCount = _layers.Count;
        foreach (var s in _scratchLocal.Values)
            s.ClearGrads();

        if (batchSize >= 8 && Environment.ProcessorCount > 1)
        {
            Parallel.For(0, batchSize, b =>
            {
                ProcessSample(batchInputs[b], batchTargets[b], stdpRate, _scratchLocal.Value!);
            });
        }
        else
        {
            var scratch = _scratchLocal.Value!;
            for (int b = 0; b < batchSize; b++)
            {
                ProcessSample(batchInputs[b], batchTargets[b], stdpRate, scratch);
            }
        }

        // Reducción de gradientes de todos los hilos
        var totalGradW = new float[layerCount][,];
        var totalGradB = new float[layerCount][];
        for (int l = 0; l < layerCount; l++)
        {
            totalGradW[l] = new float[_layers[l].OutputDim, _layers[l].InputDim];
            totalGradB[l] = new float[_layers[l].OutputDim];
        }

        float totalMse = 0f;
        long batchSpikes = 0;

        foreach (var s in _scratchLocal.Values)
        {
            totalMse += s.MseSum;
            batchSpikes += s.SpikesSum;

            for (int l = 0; l < layerCount; l++)
            {
                int outD = _layers[l].OutputDim;
                int inD = _layers[l].InputDim;
                var gw = totalGradW[l];
                var sgw = s.GradW[l];
                var gb = totalGradB[l];
                var sgb = s.GradB[l];

                for (int j = 0; j < outD; j++)
                {
                    gb[j] += sgb[j];
                    for (int i = 0; i < inD; i++)
                    {
                        gw[j, i] += sgw[j, i];
                    }
                }
            }
        }

        LastSpikesProcessed = batchSpikes / batchSize;

        // Actualización Adam respetando la máscara de topología
        _adamStep++;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        float bc1 = 1.0f - MathF.Pow(beta1, _adamStep);
        float bc2 = 1.0f - MathF.Pow(beta2, _adamStep);
        float invBatch = 1.0f / batchSize;

        for (int l = 0; l < layerCount; l++)
        {
            var layer = _layers[l];
            var gwArr = totalGradW[l];
            var gbArr = totalGradB[l];

            for (int j = 0; j < layer.OutputDim; j++)
            {
                float gb = gbArr[j] * invBatch;
                layer.BiasM[j] = beta1 * layer.BiasM[j] + (1f - beta1) * gb;
                layer.BiasV[j] = beta2 * layer.BiasV[j] + (1f - beta2) * gb * gb;
                float mHatB = layer.BiasM[j] / bc1;
                float vHatB = layer.BiasV[j] / bc2;
                layer.Biases[j] -= learningRate * mHatB / (MathF.Sqrt(vHatB) + eps);

                for (int i = 0; i < layer.InputDim; i++)
                {
                    if (!layer.Mask[j, i]) continue;

                    float gw = gwArr[j, i] * invBatch;
                    layer.AdamM[j, i] = beta1 * layer.AdamM[j, i] + (1f - beta1) * gw;
                    layer.AdamV[j, i] = beta2 * layer.AdamV[j, i] + (1f - beta2) * gw * gw;
                    float mHat = layer.AdamM[j, i] / bc1;
                    float vHat = layer.AdamV[j, i] / bc2;
                    layer.Weights[j, i] -= learningRate * mHat / (MathF.Sqrt(vHat) + eps);
                }
            }
        }

        int outCount = _layers[^1].OutputDim;
        return totalMse / (batchSize * outCount);
    }

    private void ProcessSample(float[] input, float[] target, float stdpRate, ThreadScratch scratch)
    {
        long spikes = ForwardFast(input, scratch);
        scratch.SpikesSum += spikes;

        int layerCount = _layers.Count;
        float[] pred = scratch.LayerActs[layerCount];
        int outDim = pred.Length;

        var delta = new float[outDim];
        for (int k = 0; k < outDim; k++)
        {
            float err = pred[k] - target[k];
            scratch.MseSum += err * err;
            delta[k] = (2.0f * err / outDim) * pred[k] * (1.0f - pred[k]);
        }

        for (int l = layerCount - 1; l >= 0; l--)
        {
            var layer = _layers[l];
            float[] prevAct = scratch.LayerActs[l];
            float[] prevTimes = scratch.LayerSpikeTimes[l];
            float[] nextTimes = scratch.LayerSpikeTimes[l + 1];
            var gw = scratch.GradW[l];
            var gb = scratch.GradB[l];

            float[] prevDelta = l > 0 ? new float[layer.InputDim] : Array.Empty<float>();

            for (int j = 0; j < layer.OutputDim; j++)
            {
                float dJ = delta[j];
                gb[j] += dJ;

                for (int i = 0; i < layer.InputDim; i++)
                {
                    if (!layer.Mask[j, i]) continue;

                    float g = dJ * prevAct[i];

                    if (stdpRate > 0f && !float.IsPositiveInfinity(nextTimes[j]) && !float.IsPositiveInfinity(prevTimes[i]))
                    {
                        float dt = nextTimes[j] - prevTimes[i];
                        if (dt > 0f)
                        {
                            g *= 1.0f + stdpRate * MathF.Exp(-dt / 0.25f);
                        }
                    }

                    gw[j, i] += g;

                    if (l > 0)
                    {
                        prevDelta[i] += dJ * layer.Weights[j, i];
                    }
                }
            }

            if (l > 0)
            {
                float[] prevNet = scratch.LayerNets[l - 1];
                float[] prevThresh = scratch.NeuronThreshold[l - 1];
                bool[] prevFired = scratch.NeuronFired[l - 1];

                for (int i = 0; i < prevDelta.Length; i++)
                {
                    float diff = MathF.Abs(prevNet[i] - prevThresh[i]);
                    float denom = 1.0f + diff;
                    float surrogate = (prevFired[i] ? 1.0f : 0.25f) / (denom * denom);
                    prevDelta[i] *= surrogate;
                }
                delta = prevDelta;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private long ForwardFast(ReadOnlySpan<float> input, ThreadScratch scratch)
    {
        int layerCount = _layers.Count;
        float[] inputBuf = scratch.LayerActs[0];
        float[] inputTimes = scratch.LayerSpikeTimes[0];
        Spike[] spikeBuf0 = scratch.SpikeBuffers[0];

        float maxVal = 0.00001f;
        for (int i = 0; i < input.Length; i++)
        {
            float v = input[i];
            inputBuf[i] = v;
            float abs = MathF.Abs(v);
            if (abs > maxVal) maxVal = abs;
        }

        int currentSpikeCount = 0;
        float timeWindow = _config.TimeWindow;
        for (int i = 0; i < input.Length; i++)
        {
            float val = inputBuf[i];
            float normalized = MathF.Abs(val) / maxVal;
            float t = timeWindow * (1.0f - normalized);
            inputTimes[i] = t;
            spikeBuf0[currentSpikeCount++] = new Spike(i, t, normalized, val >= 0 ? 1 : -1);
        }

        Array.Sort(spikeBuf0, 0, currentSpikeCount);
        long totalSpikes = currentSpikeCount;

        Spike[] currentSpikes = spikeBuf0;
        float[] currentValues = inputBuf;

        float baseThresh = _config.BaseThreshold;
        float decay = _config.MembraneDecay;
        float adaptRate = _config.ThresholdAdaptationRate;

        for (int l = 0; l < layerCount; l++)
        {
            var layer = _layers[l];
            bool isOutputLayer = (l == layerCount - 1);

            float[] net = scratch.LayerNets[l];
            float[] act = scratch.LayerActs[l + 1];
            float[] times = scratch.LayerSpikeTimes[l + 1];
            float[] voltages = scratch.NeuronVoltage[l];
            float[] thresholds = scratch.NeuronThreshold[l];
            bool[] fired = scratch.NeuronFired[l];

            Array.Clear(net);
            Array.Clear(voltages);
            Array.Clear(fired);
            for (int j = 0; j < layer.OutputDim; j++)
            {
                thresholds[j] = baseThresh;
                times[j] = float.PositiveInfinity;
            }

            // Propagación orientada a eventos usando lista de adyacencia dispersa
            for (int s = 0; s < currentSpikeCount; s++)
            {
                var sp = currentSpikes[s];
                int pre = sp.FeatureIndex;
                if ((uint)pre >= (uint)layer.InputDim) continue;

                float val = currentValues[pre];
                int[] targets = layer.ConnectedTargets[pre];

                for (int tIdx = 0; tIdx < targets.Length; tIdx++)
                {
                    int j = targets[tIdx];
                    float current = layer.Weights[j, pre] * val;
                    net[j] += current;

                    if (!isOutputLayer)
                    {
                        float v = (voltages[j] * decay) + current;
                        if (v >= thresholds[j])
                        {
                            if (!fired[j])
                            {
                                fired[j] = true;
                                times[j] = sp.Timestamp;
                            }
                            voltages[j] = 0.0f;
                            thresholds[j] += adaptRate;
                        }
                        else
                        {
                            voltages[j] = v;
                        }
                    }
                }
            }

            Spike[] nextSpikes = scratch.SpikeBuffers[l + 1];
            int nextSpikeCount = 0;

            for (int j = 0; j < layer.OutputDim; j++)
            {
                float z = net[j] + layer.Biases[j];
                net[j] = z;

                if (isOutputLayer)
                {
                    act[j] = 1.0f / (1.0f + MathF.Exp(-Math.Clamp(z, -20f, 20f)));
                }
                else
                {
                    float v = (voltages[j] * decay) + layer.Biases[j];
                    if (v >= thresholds[j] && !fired[j])
                    {
                        fired[j] = true;
                        times[j] = timeWindow;
                    }

                    if (fired[j]) totalSpikes++;

                    float a = z > 0f ? z / (1.0f + z) : 0.05f * z;
                    act[j] = a;

                    if (MathF.Abs(a) > 1e-4f)
                    {
                        float t = float.IsPositiveInfinity(times[j]) ? timeWindow : times[j];
                        nextSpikes[nextSpikeCount++] = new Spike(j, t, MathF.Abs(a), a >= 0 ? 1 : -1);
                    }
                }
            }

            if (!isOutputLayer && nextSpikeCount > 1)
                Array.Sort(nextSpikes, 0, nextSpikeCount);

            currentSpikes = nextSpikes;
            currentSpikeCount = nextSpikeCount;
            currentValues = act;
        }

        return totalSpikes;
    }
}

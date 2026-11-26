using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using _103KNN.Neuromorphic.Core;
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
/// Optimizada para CPU con arreglos planos 1D, SIMD (Vector&lt;float&gt;), ordenamiento inline y paralelismo por lotes.
/// </summary>
public sealed class NeuromorphicTrainableNetwork
{
    public sealed class SpikingLayerState
    {
        public int InputDim { get; }
        public int OutputDim { get; }
        public bool IsFullyConnected { get; }
        public bool[] Mask { get; }
        public int[][] ConnectedTargets { get; }
        public float[] Weights { get; }
        public float[] Biases { get; }
        public float[] AdamM { get; }
        public float[] AdamV { get; }
        public float[] BiasM { get; }
        public float[] BiasV { get; }
        public CognitiveLifNeuron[] Neurons { get; }

        public SpikingLayerState(NeuromorphicLayerMap map, CognitiveConfig config, Random rng)
        {
            InputDim = map.SourceSize;
            OutputDim = map.TargetSize;
            int total = OutputDim * InputDim;
            Mask = new bool[total];
            Weights = new float[total];
            Biases = new float[OutputDim];
            AdamM = new float[total];
            AdamV = new float[total];
            BiasM = new float[OutputDim];
            BiasV = new float[OutputDim];
            Neurons = new CognitiveLifNeuron[OutputDim];

            var perSource = new List<int>[InputDim];
            for (int i = 0; i < InputDim; i++) perSource[i] = new List<int>();

            int activeCount = 0;
            foreach (var (src, dst) in map.Connections)
            {
                if (src >= 0 && src < InputDim && dst >= 0 && dst < OutputDim)
                {
                    int idx = dst * InputDim + src;
                    if (!Mask[idx])
                    {
                        Mask[idx] = true;
                        perSource[src].Add(dst);
                        activeCount++;
                    }
                }
            }

            IsFullyConnected = activeCount == total;
            ConnectedTargets = new int[InputDim][];
            for (int i = 0; i < InputDim; i++)
                ConnectedTargets[i] = perSource[i].ToArray();

            float stdDev = MathF.Sqrt(2.0f / Math.Max(1, InputDim));
            for (int j = 0; j < OutputDim; j++)
            {
                Neurons[j] = new CognitiveLifNeuron(j, config);
                int rowOffset = j * InputDim;
                for (int i = 0; i < InputDim; i++)
                {
                    int idx = rowOffset + i;
                    if (Mask[idx])
                    {
                        Weights[idx] = (float)((rng.NextDouble() * 2.0 - 1.0) * stdDev);
                    }
                }
            }
        }
    }

    private sealed class ThreadScratch
    {
        public readonly float[][] GradW;
        public readonly float[][] GradB;
        public readonly float[][] LayerActs;
        public readonly float[][] LayerNets;
        public readonly float[][] LayerSpikeTimes;
        public readonly float[][] NeuronVoltage;
        public readonly float[][] NeuronThreshold;
        public readonly bool[][] NeuronFired;
        public readonly Spike[][] SpikeBuffers;
        public readonly float[][] DeltaBuffers;
        public float MseSum;
        public long SpikesSum;

        public ThreadScratch(IReadOnlyList<SpikingLayerState> layers)
        {
            int count = layers.Count;
            GradW = new float[count][];
            GradB = new float[count][];
            LayerActs = new float[count + 1][];
            LayerNets = new float[count][];
            LayerSpikeTimes = new float[count + 1][];
            NeuronVoltage = new float[count][];
            NeuronThreshold = new float[count][];
            NeuronFired = new bool[count][];
            SpikeBuffers = new Spike[count + 1][];
            DeltaBuffers = new float[count + 1][];

            LayerActs[0] = new float[layers[0].InputDim];
            LayerSpikeTimes[0] = new float[layers[0].InputDim];
            SpikeBuffers[0] = new Spike[layers[0].InputDim];
            DeltaBuffers[0] = new float[layers[0].InputDim];

            for (int l = 0; l < count; l++)
            {
                int inD = layers[l].InputDim;
                int outD = layers[l].OutputDim;
                GradW[l] = new float[outD * inD];
                GradB[l] = new float[outD];
                LayerActs[l + 1] = new float[outD];
                LayerNets[l] = new float[outD];
                LayerSpikeTimes[l + 1] = new float[outD];
                NeuronVoltage[l] = new float[outD];
                NeuronThreshold[l] = new float[outD];
                NeuronFired[l] = new bool[outD];
                SpikeBuffers[l + 1] = new Spike[outD];
                DeltaBuffers[l + 1] = new float[outD];
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
    private readonly float[][] _totalGradW;
    private readonly float[][] _totalGradB;
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

        _totalGradW = new float[_layers.Count][];
        _totalGradB = new float[_layers.Count][];
        for (int l = 0; l < _layers.Count; l++)
        {
            _totalGradW[l] = new float[_layers[l].OutputDim * _layers[l].InputDim];
            _totalGradB[l] = new float[_layers[l].OutputDim];
        }

        _scratchLocal = new ThreadLocal<ThreadScratch>(() => new ThreadScratch(_layers), trackAllValues: true);
    }

    public float[] Predict(ReadOnlySpan<float> input)
    {
        var scratch = _scratchLocal.Value!;
        long spikes = ForwardFast(input, scratch);
        LastSpikesProcessed = spikes;
        return (float[])scratch.LayerActs[_layers.Count].Clone();
    }

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

        for (int l = 0; l < layerCount; l++)
        {
            Array.Clear(_totalGradW[l]);
            Array.Clear(_totalGradB[l]);
        }

        float totalMse = 0f;
        long batchSpikes = 0;
        int vecLen = Vector<float>.Count;

        foreach (var s in _scratchLocal.Values)
        {
            if (s.SpikesSum == 0 && s.MseSum == 0f) continue;
            totalMse += s.MseSum;
            batchSpikes += s.SpikesSum;

            for (int l = 0; l < layerCount; l++)
            {
                float[] gw = _totalGradW[l];
                float[] sgw = s.GradW[l];
                int len = gw.Length;
                int idx = 0;
                for (; idx <= len - vecLen; idx += vecLen)
                {
                    var v = new Vector<float>(gw, idx) + new Vector<float>(sgw, idx);
                    v.CopyTo(gw, idx);
                }
                for (; idx < len; idx++) gw[idx] += sgw[idx];

                float[] gb = _totalGradB[l];
                float[] sgb = s.GradB[l];
                for (int j = 0; j < gb.Length; j++) gb[j] += sgb[j];
            }
        }

        LastSpikesProcessed = batchSpikes / batchSize;

        _adamStep++;
        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        float bc1 = 1.0f - MathF.Pow(beta1, _adamStep);
        float bc2 = 1.0f - MathF.Pow(beta2, _adamStep);
        float invBatch = 1.0f / batchSize;
        float lrCorrected = learningRate * MathF.Sqrt(bc2) / bc1;

        for (int l = 0; l < layerCount; l++)
        {
            var layer = _layers[l];
            float[] gwArr = _totalGradW[l];
            float[] gbArr = _totalGradB[l];
            float[] wArr = layer.Weights;
            float[] mArr = layer.AdamM;
            float[] vArr = layer.AdamV;
            bool[] maskArr = layer.Mask;
            bool full = layer.IsFullyConnected;

            for (int j = 0; j < layer.OutputDim; j++)
            {
                float gb = gbArr[j] * invBatch;
                layer.BiasM[j] = beta1 * layer.BiasM[j] + (1f - beta1) * gb;
                layer.BiasV[j] = beta2 * layer.BiasV[j] + (1f - beta2) * gb * gb;
                float mHatB = layer.BiasM[j] / bc1;
                float vHatB = layer.BiasV[j] / bc2;
                layer.Biases[j] -= learningRate * mHatB / (MathF.Sqrt(vHatB) + eps);
            }

            int totalW = wArr.Length;
            if (full && Vector.IsHardwareAccelerated && totalW >= vecLen)
            {
                var vInvBatch = new Vector<float>(invBatch);
                var vB1 = new Vector<float>(beta1);
                var vOneMinusB1 = new Vector<float>(1f - beta1);
                var vB2 = new Vector<float>(beta2);
                var vOneMinusB2 = new Vector<float>(1f - beta2);
                var vLr = new Vector<float>(lrCorrected);
                var vEps = new Vector<float>(eps);

                int idx = 0;
                for (; idx <= totalW - vecLen; idx += vecLen)
                {
                    var g = new Vector<float>(gwArr, idx) * vInvBatch;
                    var m = vB1 * new Vector<float>(mArr, idx) + vOneMinusB1 * g;
                    var v = vB2 * new Vector<float>(vArr, idx) + vOneMinusB2 * g * g;
                    m.CopyTo(mArr, idx);
                    v.CopyTo(vArr, idx);

                    var w = new Vector<float>(wArr, idx) - vLr * m / (Vector.SquareRoot(v) + vEps);
                    w.CopyTo(wArr, idx);
                }
                for (; idx < totalW; idx++)
                {
                    float gw = gwArr[idx] * invBatch;
                    mArr[idx] = beta1 * mArr[idx] + (1f - beta1) * gw;
                    vArr[idx] = beta2 * vArr[idx] + (1f - beta2) * gw * gw;
                    wArr[idx] -= lrCorrected * mArr[idx] / (MathF.Sqrt(vArr[idx]) + eps);
                }
            }
            else
            {
                for (int idx = 0; idx < totalW; idx++)
                {
                    if (!full && !maskArr[idx]) continue;
                    float gw = gwArr[idx] * invBatch;
                    mArr[idx] = beta1 * mArr[idx] + (1f - beta1) * gw;
                    vArr[idx] = beta2 * vArr[idx] + (1f - beta2) * gw * gw;
                    wArr[idx] -= lrCorrected * mArr[idx] / (MathF.Sqrt(vArr[idx]) + eps);
                }
            }
        }

        int outCount = _layers[^1].OutputDim;
        return totalMse / (batchSize * outCount);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void ProcessSample(float[] input, float[] target, float stdpRate, ThreadScratch scratch)
    {
        long spikes = ForwardFast(input, scratch);
        scratch.SpikesSum += spikes;

        int layerCount = _layers.Count;
        float[] pred = scratch.LayerActs[layerCount];
        int outDim = pred.Length;

        float[] delta = scratch.DeltaBuffers[layerCount];
        for (int k = 0; k < outDim; k++)
        {
            float err = pred[k] - target[k];
            scratch.MseSum += err * err;
            float sig = Math.Clamp((pred[k] + 0.1f) / 1.2f, 1e-4f, 1.0f - 1e-4f);
            delta[k] = (2.0f * err / outDim) * (1.2f * sig * (1.0f - sig));
        }

        int vecLen = Vector<float>.Count;

        for (int l = layerCount - 1; l >= 0; l--)
        {
            var layer = _layers[l];
            int inDim = layer.InputDim;
            int outD = layer.OutputDim;
            float[] prevAct = scratch.LayerActs[l];
            float[] prevTimes = scratch.LayerSpikeTimes[l];
            float[] nextTimes = scratch.LayerSpikeTimes[l + 1];
            float[] gw = scratch.GradW[l];
            float[] gb = scratch.GradB[l];
            float[] wArr = layer.Weights;
            bool[] maskArr = layer.Mask;
            bool full = layer.IsFullyConnected;

            float[] prevDelta = scratch.DeltaBuffers[l];
            if (l > 0) Array.Clear(prevDelta);

            for (int j = 0; j < outD; j++)
            {
                float dJ = delta[j];
                gb[j] += dJ;
                int rowOffset = j * inDim;

                if (full && stdpRate <= 0f && Vector.IsHardwareAccelerated && inDim >= vecLen)
                {
                    var vDj = new Vector<float>(dJ);
                    int i = 0;
                    for (; i <= inDim - vecLen; i += vecLen)
                    {
                        var vAct = new Vector<float>(prevAct, i);
                        var vGw = new Vector<float>(gw, rowOffset + i) + vDj * vAct;
                        vGw.CopyTo(gw, rowOffset + i);

                        if (l > 0)
                        {
                            var vPd = new Vector<float>(prevDelta, i) + vDj * new Vector<float>(wArr, rowOffset + i);
                            vPd.CopyTo(prevDelta, i);
                        }
                    }
                    for (; i < inDim; i++)
                    {
                        int idx = rowOffset + i;
                        gw[idx] += dJ * prevAct[i];
                        if (l > 0) prevDelta[i] += dJ * wArr[idx];
                    }
                }
                else
                {
                    float tPost = nextTimes[j];
                    bool hasPost = stdpRate > 0f && !float.IsPositiveInfinity(tPost);

                    for (int i = 0; i < inDim; i++)
                    {
                        int idx = rowOffset + i;
                        if (!full && !maskArr[idx]) continue;

                        float g = dJ * prevAct[i];
                        if (hasPost && !float.IsPositiveInfinity(prevTimes[i]))
                        {
                            float dt = tPost - prevTimes[i];
                            if (dt > 0f)
                                g *= 1.0f + stdpRate * MathF.Exp(-dt / 0.25f);
                        }

                        gw[idx] += g;
                        if (l > 0) prevDelta[i] += dJ * wArr[idx];
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
                    float z = prevNet[i];
                    float surrogate = prevFired[i]
                        ? (z > 0f ? 1.0f : 0.1f)
                        : (0.25f / (1.0f + 0.5f * MathF.Abs(z - prevThresh[i])));
                    prevDelta[i] *= surrogate;
                }
                delta = prevDelta;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SortSpikesInline(Spike[] arr, int count)
    {
        for (int i = 1; i < count; i++)
        {
            Spike key = arr[i];
            float keyT = key.Timestamp;
            int j = i - 1;
            while (j >= 0 && arr[j].Timestamp > keyT)
            {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
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

        if (currentSpikeCount > 1) SortSpikesInline(spikeBuf0, currentSpikeCount);
        long totalSpikes = currentSpikeCount;

        Spike[] currentSpikes = spikeBuf0;
        float[] currentValues = inputBuf;

        float baseThresh = _config.BaseThreshold;
        float decay = _config.MembraneDecay;
        float adaptRate = _config.ThresholdAdaptationRate;

        for (int l = 0; l < layerCount; l++)
        {
            var layer = _layers[l];
            int inDim = layer.InputDim;
            int outDim = layer.OutputDim;
            bool isOutputLayer = (l == layerCount - 1);

            float[] net = scratch.LayerNets[l];
            float[] act = scratch.LayerActs[l + 1];
            float[] times = scratch.LayerSpikeTimes[l + 1];
            float[] voltages = scratch.NeuronVoltage[l];
            float[] thresholds = scratch.NeuronThreshold[l];
            bool[] fired = scratch.NeuronFired[l];
            float[] wArr = layer.Weights;

            Array.Clear(net);
            Array.Clear(voltages);
            Array.Clear(fired);
            for (int j = 0; j < outDim; j++)
            {
                thresholds[j] = baseThresh;
                times[j] = float.PositiveInfinity;
            }

            for (int s = 0; s < currentSpikeCount; s++)
            {
                var sp = currentSpikes[s];
                int pre = sp.FeatureIndex;
                if ((uint)pre >= (uint)inDim) continue;

                float val = currentValues[pre];
                int[] targets = layer.ConnectedTargets[pre];

                for (int tIdx = 0; tIdx < targets.Length; tIdx++)
                {
                    int j = targets[tIdx];
                    float current = wArr[j * inDim + pre] * val;
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

            for (int j = 0; j < outDim; j++)
            {
                float z = net[j] + layer.Biases[j];
                net[j] = z;

                if (isOutputLayer)
                {
                    float sig = 1.0f / (1.0f + MathF.Exp(-Math.Clamp(z, -20f, 20f)));
                    act[j] = 1.2f * sig - 0.1f;
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

                    float a = fired[j]
                        ? (z > 0f ? z : 0.1f * z)
                        : (z > 0f ? 0.25f * z : 0.05f * z);
                    act[j] = a;

                    if (MathF.Abs(a) > 1e-4f)
                    {
                        float t = float.IsPositiveInfinity(times[j]) ? timeWindow : times[j];
                        nextSpikes[nextSpikeCount++] = new Spike(j, t, MathF.Abs(a), a >= 0 ? 1 : -1);
                    }
                }
            }

            if (!isOutputLayer && nextSpikeCount > 1)
                SortSpikesInline(nextSpikes, nextSpikeCount);

            currentSpikes = nextSpikes;
            currentSpikeCount = nextSpikeCount;
            currentValues = act;
        }

        return totalSpikes;
    }
}

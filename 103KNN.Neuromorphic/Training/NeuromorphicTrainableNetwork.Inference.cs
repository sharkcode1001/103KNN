using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading;
using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Training;

public sealed partial class NeuromorphicTrainableNetwork
{
    private float[] PredictInternal(ReadOnlySpan<float> input, out NeuromorphicInferenceMetrics metrics)
    {
        var scratch = _scratchLocal.Value!;
        long spikes = ForwardFast(input, scratch, out metrics);
        Interlocked.Exchange(ref _lastSpikesProcessed, spikes);
        Interlocked.Exchange(ref _lastLifFiredCount, metrics.LifFiredCount);
        Interlocked.Exchange(ref _lastPropagatedEventCount, metrics.PropagatedEventCount);
        Interlocked.Exchange(ref _lastSynapticOpsCount, metrics.SynapticOpsCount);
        return (float[])scratch.LayerActs[_layers.Count].Clone();
    }

    public float[] Predict(ReadOnlySpan<float> input) => Predict(input, out _);

    /// <summary>
    /// Ejecuta inferencia devolviendo las métricas independientes de esta llamada concreta.
    /// </summary>
    public float[] Predict(ReadOnlySpan<float> input, out NeuromorphicInferenceMetrics metrics)
    {
        ThrowIfDisposed();
        int expectedIn = _layers[0].InputDim;
        if (input.Length != expectedIn)
            throw new ArgumentException($"Dimensión de entrada inválida: se esperaban {expectedIn} valores y se recibieron {input.Length}.", nameof(input));
        ValidateFiniteValues(input, nameof(input));

        EnterReadLockSafe();
        try
        {
            return PredictInternal(input, out metrics);
        }
        finally
        {
            ExitReadLockSafe();
        }
    }

    /// <summary>
    /// Predice en escala original aplicando automáticamente los rangos Min/Max guardados en el modelo.
    /// </summary>
    public float[] PredictRaw(ReadOnlySpan<float> rawInput) => PredictRaw(rawInput, out _);

    /// <summary>
    /// Predice en escala original devolviendo las métricas independientes de esta llamada concreta.
    /// </summary>
    public float[] PredictRaw(ReadOnlySpan<float> rawInput, out NeuromorphicInferenceMetrics metrics)
    {
        ThrowIfDisposed();
        int expectedIn = _layers[0].InputDim;
        int expectedOut = _layers[^1].OutputDim;
        if (rawInput.Length != expectedIn)
            throw new ArgumentException($"Dimensión de entrada inválida: se esperaban {expectedIn} valores y se recibieron {rawInput.Length}.", nameof(rawInput));
        ValidateFiniteValues(rawInput, nameof(rawInput));

        if ((MinIn == null) != (MaxIn == null))
            throw new InvalidOperationException("MinIn y MaxIn deben estar ambos definidos o ambos nulos.");
        if (MinIn != null && (MinIn.Length != expectedIn || MaxIn!.Length != expectedIn))
            throw new InvalidOperationException($"Las escalas MinIn/MaxIn deben tener longitud {expectedIn}.");

        if ((MinOut == null) != (MaxOut == null))
            throw new InvalidOperationException("MinOut y MaxOut deben estar ambos definidos o ambos nulos.");
        if (MinOut != null && (MinOut.Length != expectedOut || MaxOut!.Length != expectedOut))
            throw new InvalidOperationException($"Las escalas MinOut/MaxOut deben tener longitud {expectedOut}.");

        const int StackAllocThreshold = 256;
        float[]? rented = expectedIn > StackAllocThreshold ? ArrayPool<float>.Shared.Rent(expectedIn) : null;
        try
        {
            Span<float> scaled = rented != null
                ? rented.AsSpan(0, expectedIn)
                : stackalloc float[expectedIn];

            if (MinIn != null && MaxIn != null)
            {
                for (int i = 0; i < expectedIn; i++)
                {
                    float range = MaxIn[i] - MinIn[i];
                    scaled[i] = range == 0f ? 0f : (rawInput[i] - MinIn[i]) / range;
                }
            }
            else
            {
                rawInput.CopyTo(scaled);
            }

            float[] pred;
            EnterReadLockSafe();
            try
            {
                pred = PredictInternal(scaled, out metrics);
            }
            finally
            {
                ExitReadLockSafe();
            }

            if (MinOut != null && MaxOut != null)
            {
                for (int k = 0; k < expectedOut; k++)
                {
                    float range = MaxOut[k] - MinOut[k];
                    pred[k] = (pred[k] * range) + MinOut[k];
                }
            }
            return pred;
        }
        finally
        {
            if (rented != null)
                ArrayPool<float>.Shared.Return(rented);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ComputeTimeDecay(float dt, float decay, float logDecay)
    {
        if (dt <= 0f || decay >= 1f) return 1f;
        if (decay <= 0f) return 0f;
        return MathF.Exp(dt * logDecay);
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

    /// <summary>
    /// Camino rápido de propagación hacia adelante sobre arreglos planos 1D en <see cref="ThreadScratch"/>.
    /// </summary>
    /// <remarks>
    /// Diferencias respecto a <see cref="Neurons.CognitiveLifNeuron"/>:
    /// <list type="bullet">
    /// <item><description>Opera directamente sobre buffers planos (<c>voltages</c>, <c>thresholds</c>, <c>lastTimes</c>, <c>fired</c>) sin invocar las instancias de <see cref="SpikingLayerState.Neurons"/>.</description></item>
    /// <item><description>No aplica inhibición lateral entre neuronas competidoras ni el paso discreto <see cref="Neurons.CognitiveLifNeuron.Step"/>.</description></item>
    /// <item><description>Evalúa el disparo LIF y el cierre de ventana temporal (<c>timeWindow</c> + sesgo) sobre <c>voltages[j]</c>, mientras que acumula <c>net[j]</c> de forma lineal para calcular la amplitud propagada y el surrogate gradient.</description></item>
    /// </list>
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private long ForwardFast(ReadOnlySpan<float> input, ThreadScratch scratch, out NeuromorphicInferenceMetrics metrics)
    {
        int layerCount = _layers.Count;
        float[] inputBuf = scratch.LayerActs[0];
        float[] inputTimes = scratch.LayerSpikeTimes[0];
        Spike[] spikeBuf0 = scratch.SpikeBuffers[0];

        bool useGlobalScale = _config.EncodingMode == TimeToFirstSpikeMode.GlobalScale;
        float maxVal = useGlobalScale ? _config.GlobalEncodingScale : 0.00001f;
        for (int i = 0; i < input.Length; i++)
        {
            float v = input[i];
            inputBuf[i] = v;
            if (!useGlobalScale)
            {
                float abs = MathF.Abs(v);
                if (abs > maxVal) maxVal = abs;
            }
        }

        int currentSpikeCount = 0;
        float timeWindow = _config.TimeWindow;
        for (int i = 0; i < input.Length; i++)
        {
            float val = inputBuf[i];
            if (val == 0f)
            {
                inputTimes[i] = float.PositiveInfinity;
                continue;
            }
            float normalized = Math.Clamp(MathF.Abs(val) / maxVal, 0f, 1f);
            float t = timeWindow * (1.0f - normalized);
            inputTimes[i] = t;
            spikeBuf0[currentSpikeCount++] = new Spike(i, t, normalized, val >= 0 ? 1 : -1);
        }

        if (currentSpikeCount > 1) SortSpikesInline(spikeBuf0, currentSpikeCount);
        long totalSpikes = currentSpikeCount;
        long lifFiredCount = 0;
        long propagatedEventCount = currentSpikeCount;
        long synapticOpsCount = 0;

        Spike[] currentSpikes = spikeBuf0;
        float[] currentValues = inputBuf;

        float baseThresh = _config.BaseThreshold;
        float decay = _config.MembraneDecay;
        float threshDecay = _config.ThresholdDecay;
        float adaptRate = _config.ThresholdAdaptationRate;
        float logDecay = decay > 0f ? MathF.Log(decay) : float.NegativeInfinity;
        float logThreshDecay = threshDecay > 0f ? MathF.Log(threshDecay) : float.NegativeInfinity;
        bool strictSpiking = _config.ExecutionMode == NeuromorphicExecutionMode.StrictSpiking;

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
            float[] lastTimes = scratch.NeuronLastTimestamp[l];
            bool[] fired = scratch.NeuronFired[l];
            float[] wArr = layer.Weights;

            Array.Clear(net);
            Array.Clear(voltages);
            Array.Clear(lastTimes);
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
                synapticOpsCount += targets.Length;

                for (int tIdx = 0; tIdx < targets.Length; tIdx++)
                {
                    int j = targets[tIdx];
                    float current = wArr[j * inDim + pre] * val;
                    net[j] += current;

                    if (!isOutputLayer)
                    {
                        float dt = sp.Timestamp - lastTimes[j];
                        if (dt > 0f)
                        {
                            voltages[j] *= ComputeTimeDecay(dt, decay, logDecay);
                            thresholds[j] = baseThresh + (thresholds[j] - baseThresh) * ComputeTimeDecay(dt, threshDecay, logThreshDecay);
                            lastTimes[j] = sp.Timestamp;
                        }

                        float v = voltages[j] + current;
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
                    float dtEnd = timeWindow - lastTimes[j];
                    float endLeak = dtEnd > 0f ? ComputeTimeDecay(dtEnd, decay, logDecay) : 1.0f;
                    float v = (voltages[j] * endLeak) + layer.Biases[j];
                    if (v >= thresholds[j] && !fired[j])
                    {
                        fired[j] = true;
                        times[j] = timeWindow;
                    }

                    if (fired[j])
                    {
                        totalSpikes++;
                        lifFiredCount++;
                    }

                    if (strictSpiking)
                    {
                        float a = fired[j] ? (z > 0f ? z : 0.1f * z) : 0.0f;
                        act[j] = a;

                        if (fired[j] && MathF.Abs(a) > 1e-4f)
                        {
                            nextSpikes[nextSpikeCount++] = new Spike(j, times[j], MathF.Abs(a), a >= 0 ? 1 : -1);
                        }
                    }
                    else
                    {
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
            }

            if (!isOutputLayer && nextSpikeCount > 1)
                SortSpikesInline(nextSpikes, nextSpikeCount);

            if (!isOutputLayer)
                propagatedEventCount += nextSpikeCount;

            currentSpikes = nextSpikes;
            currentSpikeCount = nextSpikeCount;
            currentValues = act;
        }

        metrics = new NeuromorphicInferenceMetrics(lifFiredCount, propagatedEventCount, synapticOpsCount);
        return totalSpikes;
    }
}

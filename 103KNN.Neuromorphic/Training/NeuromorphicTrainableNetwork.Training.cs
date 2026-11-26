using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace _103KNN.Neuromorphic.Training;

public sealed partial class NeuromorphicTrainableNetwork
{
    public float TrainStep(float[][] batchInputs, float[][] batchTargets, float learningRate = 0.005f, float stdpRate = 0.002f)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(batchInputs);
        ArgumentNullException.ThrowIfNull(batchTargets);
        if (batchInputs.Length != batchTargets.Length)
            throw new ArgumentException($"batchInputs ({batchInputs.Length}) y batchTargets ({batchTargets.Length}) deben tener la misma cantidad de muestras.");
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(learningRate), learningRate, "learningRate debe ser finito y mayor que 0.");
        if (!float.IsFinite(stdpRate) || stdpRate < 0f)
            throw new ArgumentOutOfRangeException(nameof(stdpRate), stdpRate, "stdpRate debe ser finito y >= 0.");

        int batchSize = batchInputs.Length;
        if (batchSize == 0) return 0f;

        int expectedIn = _layers[0].InputDim;
        int expectedOut = _layers[^1].OutputDim;
        for (int b = 0; b < batchSize; b++)
        {
            if (batchInputs[b] == null)
                throw new ArgumentNullException(nameof(batchInputs), $"La muestra de entrada {b} es nula.");
            if (batchTargets[b] == null)
                throw new ArgumentNullException(nameof(batchTargets), $"La muestra objetivo {b} es nula.");
            if (batchInputs[b].Length != expectedIn)
                throw new ArgumentException($"La muestra de entrada {b} tiene {batchInputs[b].Length} valores; se esperaban {expectedIn}.", nameof(batchInputs));
            if (batchTargets[b].Length != expectedOut)
                throw new ArgumentException($"La muestra objetivo {b} tiene {batchTargets[b].Length} valores; se esperaban {expectedOut}.", nameof(batchTargets));
            ValidateFiniteValues(batchInputs[b], nameof(batchInputs));
            ValidateFiniteValues(batchTargets[b], nameof(batchTargets));
        }

        EnterWriteLockSafe();
        try
        {
            return TrainStepCore(batchInputs, batchTargets, learningRate, stdpRate);
        }
        finally
        {
            ExitWriteLockSafe();
        }
    }

    private float TrainStepCore(float[][] batchInputs, float[][] batchTargets, float learningRate, float stdpRate)
    {
        int batchSize = batchInputs.Length;
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
        long batchLifFired = 0;
        long batchPropagated = 0;
        long batchSynapticOps = 0;
        int vecLen = Vector<float>.Count;

        foreach (var s in _scratchLocal.Values)
        {
            if (s.SpikesSum == 0 && s.MseSum == 0f) continue;
            totalMse += s.MseSum;
            batchSpikes += s.SpikesSum;
            batchLifFired += s.LifFiredSum;
            batchPropagated += s.PropagatedEventsSum;
            batchSynapticOps += s.SynapticOpsSum;

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

        Interlocked.Exchange(ref _lastSpikesProcessed, batchSpikes / batchSize);
        Interlocked.Exchange(ref _lastLifFiredCount, batchLifFired / batchSize);
        Interlocked.Exchange(ref _lastPropagatedEventCount, batchPropagated / batchSize);
        Interlocked.Exchange(ref _lastSynapticOpsCount, batchSynapticOps / batchSize);

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
        long spikes = ForwardFast(input, scratch, out var sampleMetrics);
        scratch.SpikesSum += spikes;
        scratch.LifFiredSum += sampleMetrics.LifFiredCount;
        scratch.PropagatedEventsSum += sampleMetrics.PropagatedEventCount;
        scratch.SynapticOpsSum += sampleMetrics.SynapticOpsCount;

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
                            else if (dt < 0f && _config.EnableStdpDepression)
                                g *= MathF.Max(0.1f, 1.0f - stdpRate * 0.5f * MathF.Exp(dt / 0.25f));
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
}

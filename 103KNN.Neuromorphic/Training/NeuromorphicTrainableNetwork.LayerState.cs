using System;
using System.Collections.Generic;
using System.Threading;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Neurons;

namespace _103KNN.Neuromorphic.Training;

public sealed partial class NeuromorphicTrainableNetwork
{
    public sealed class SpikingLayerState
    {
        private readonly CognitiveConfig _config;
        private CognitiveLifNeuron[]? _neurons;

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
        public CognitiveLifNeuron[] Neurons => GetOrCreateNeurons();

        private CognitiveLifNeuron[] GetOrCreateNeurons()
        {
            var existing = Volatile.Read(ref _neurons);
            if (existing != null)
                return existing;

            var created = new CognitiveLifNeuron[OutputDim];
            for (int j = 0; j < OutputDim; j++)
                created[j] = new CognitiveLifNeuron(j, _config);

            return Interlocked.CompareExchange(ref _neurons, created, null) ?? created;
        }

        public SpikingLayerState(NeuromorphicLayerMap map, CognitiveConfig config, Random rng)
        {
            ArgumentNullException.ThrowIfNull(map);
            ArgumentNullException.ThrowIfNull(config);
            if (map.SourceSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(map), map.SourceSize, "SourceSize de la capa debe ser mayor que 0.");
            if (map.TargetSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(map), map.TargetSize, "TargetSize de la capa debe ser mayor que 0.");

            _config = config;
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

            var perSource = new List<int>[InputDim];
            for (int i = 0; i < InputDim; i++) perSource[i] = new List<int>();

            int activeCount = 0;
            foreach (var (src, dst) in map.Connections)
            {
                if (src < 0 || src >= InputDim)
                    throw new ArgumentOutOfRangeException(nameof(map), src, $"Conexión inválida: origen {src} fuera de [0, {InputDim}).");
                if (dst < 0 || dst >= OutputDim)
                    throw new ArgumentOutOfRangeException(nameof(map), dst, $"Conexión inválida: destino {dst} fuera de [0, {OutputDim}).");

                int idx = dst * InputDim + src;
                if (!Mask[idx])
                {
                    Mask[idx] = true;
                    perSource[src].Add(dst);
                    activeCount++;
                }
            }

            IsFullyConnected = activeCount == total;
            ConnectedTargets = new int[InputDim][];
            for (int i = 0; i < InputDim; i++)
                ConnectedTargets[i] = perSource[i].ToArray();

            float stdDev = MathF.Sqrt(2.0f / Math.Max(1, InputDim));
            for (int j = 0; j < OutputDim; j++)
            {
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
}

using System;
using System.Collections.Generic;
using _103KNN.Neuromorphic.Core;

namespace _103KNN.Neuromorphic.Training;

public sealed partial class NeuromorphicTrainableNetwork
{
    private sealed class ThreadScratch
    {
        public readonly float[][] GradW;
        public readonly float[][] GradB;
        public readonly float[][] LayerActs;
        public readonly float[][] LayerNets;
        public readonly float[][] LayerSpikeTimes;
        public readonly float[][] NeuronVoltage;
        public readonly float[][] NeuronThreshold;
        public readonly float[][] NeuronLastTimestamp;
        public readonly bool[][] NeuronFired;
        public readonly Spike[][] SpikeBuffers;
        public readonly float[][] DeltaBuffers;
        public float MseSum;
        public long SpikesSum;
        public long LifFiredSum;
        public long PropagatedEventsSum;
        public long SynapticOpsSum;

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
            NeuronLastTimestamp = new float[count][];
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
                NeuronLastTimestamp[l] = new float[outD];
                NeuronFired[l] = new bool[outD];
                SpikeBuffers[l + 1] = new Spike[outD];
                DeltaBuffers[l + 1] = new float[outD];
            }
        }

        public void ClearGrads()
        {
            MseSum = 0f;
            SpikesSum = 0;
            LifFiredSum = 0;
            PropagatedEventsSum = 0;
            SynapticOpsSum = 0;
            for (int l = 0; l < GradW.Length; l++)
            {
                Array.Clear(GradW[l]);
                Array.Clear(GradB[l]);
            }
        }
    }
}

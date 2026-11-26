using System;
using System.Collections.Generic;

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
        if (sourceSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceSize), sourceSize, "El tamaño de origen debe ser mayor que 0.");
        if (targetSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), targetSize, "El tamaño de destino debe ser mayor que 0.");

        SourceSize = sourceSize;
        TargetSize = targetSize;
        if (connections != null)
        {
            foreach (var conn in connections)
            {
                if (conn.Source < 0 || conn.Source >= sourceSize)
                    throw new ArgumentOutOfRangeException(nameof(connections), conn.Source, $"Índice de origen {conn.Source} fuera de rango [0, {sourceSize}).");
                if (conn.Target < 0 || conn.Target >= targetSize)
                    throw new ArgumentOutOfRangeException(nameof(connections), conn.Target, $"Índice de destino {conn.Target} fuera de rango [0, {targetSize}).");
                Connections.Add(conn);
            }
        }
        else
        {
            for (int s = 0; s < sourceSize; s++)
                for (int t = 0; t < targetSize; t++)
                    Connections.Add((s, t));
        }
    }
}

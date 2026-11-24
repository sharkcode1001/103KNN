namespace _103KNN.Neuromorphic.Associative;

/// <summary>
/// Muestra o prototipo etiquetado para el clasificador k-NN asociativo.
/// </summary>
public sealed record KnnSample(
    int Id,
    int ClassLabel,
    float[] Features
);

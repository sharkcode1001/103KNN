namespace _103KNN.Neuromorphic.Core;

/// <summary>
/// Representa un evento discreto (espiga) en el dominio temporal.
/// </summary>
public readonly record struct Spike(
    int FeatureIndex,
    float Timestamp,
    float Magnitude = 1.0f,
    int Polarity = 1
) : IComparable<Spike>
{
    public int CompareTo(Spike other) => Timestamp.CompareTo(other.Timestamp);
}

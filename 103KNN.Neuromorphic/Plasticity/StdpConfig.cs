namespace _103KNN.Neuromorphic.Plasticity;

/// <summary>
/// Parámetros de la regla biológica STDP (Spike-Timing-Dependent Plasticity).
/// Modela la potenciación y depresión a largo plazo según la diferencia temporal de espigas (delta t = t_post - t_pre).
/// </summary>
public sealed class StdpConfig
{
    /// <summary>
    /// Amplitud de potenciación a largo plazo (LTP - Long Term Potentiation).
    /// </summary>
    public float APlus { get; init; } = 0.025f;

    /// <summary>
    /// Amplitud de depresión a largo plazo (LTD - Long Term Depression).
    /// </summary>
    public float AMinus { get; init; } = 0.020f;

    /// <summary>
    /// Constante de tiempo para LTP (en unidades temporales del simulador).
    /// </summary>
    public float TauPlus { get; init; } = 0.20f;

    /// <summary>
    /// Constante de tiempo para LTD.
    /// </summary>
    public float TauMinus { get; init; } = 0.25f;

    /// <summary>
    /// Peso sináptico mínimo permitido.
    /// </summary>
    public float MinWeight { get; init; } = -2.0f;

    /// <summary>
    /// Peso sináptico máximo permitido.
    /// </summary>
    public float MaxWeight { get; init; } = 2.0f;

    /// <summary>
    /// Tasa de aprendizaje / escala de actualización.
    /// </summary>
    public float LearningRate { get; init; } = 1.0f;
}

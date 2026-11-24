namespace _103KNN.Neuromorphic.Core;

/// <summary>
/// Configuración de hiperparámetros para la dinámica cognitiva neuromórfica.
/// </summary>
public sealed class CognitiveConfig
{
    /// <summary>
    /// Umbral base de disparo de membrana (V_th).
    /// </summary>
    public float BaseThreshold { get; init; } = 0.35f;

    /// <summary>
    /// Factor de decaimiento temporal de membrana por paso (lambda / leak).
    /// </summary>
    public float MembraneDecay { get; init; } = 0.90f;

    /// <summary>
    /// Coeficiente de inhibición lateral aplicado a neuronas vecinas/competidoras.
    /// </summary>
    public float LateralInhibitionFactor { get; init; } = 0.15f;

    /// <summary>
    /// Incremento adaptativo del umbral tras disparar (refractario / homeostasis).
    /// </summary>
    public float ThresholdAdaptationRate { get; init; } = 0.20f;

    /// <summary>
    /// Decaimiento del umbral adaptativo hacia el umbral base.
    /// </summary>
    public float ThresholdDecay { get; init; } = 0.95f;

    /// <summary>
    /// Número máximo de pasos temporales para deliberación profunda (System 2).
    /// </summary>
    public int MaxDeliberationSteps { get; init; } = 25;

    /// <summary>
    /// Margen relativo de disparo para activar salida temprana (System 1 - Fast Thinking).
    /// </summary>
    public float EarlyExitConfidenceThreshold { get; init; } = 0.75f;

    /// <summary>
    /// Ventana de tiempo máxima para codificación temporal (TTFS).
    /// </summary>
    public float TimeWindow { get; init; } = 1.0f;
}

namespace _103KNN.Neuromorphic.Core;

/// <summary>
/// Modo de ejecución del motor neuromórfico.
/// </summary>
public enum NeuromorphicExecutionMode
{
    /// <summary>
    /// Modo híbrido: combina disparos LIF con propagación de activaciones continuas sub-umbral.
    /// </summary>
    Hybrid = 0,

    /// <summary>
    /// Modo SNN estricto: únicamente las neuronas que superan el umbral LIF propagan espigas.
    /// </summary>
    StrictSpiking = 1
}

/// <summary>
/// Modo de normalización temporal para el codificador Time-To-First-Spike (TTFS).
/// </summary>
public enum TimeToFirstSpikeMode
{
    /// <summary>
    /// Normaliza por el máximo absoluto de cada muestra individual.
    /// </summary>
    PerSampleRelative = 0,

    /// <summary>
    /// Normaliza usando una escala global fija (<see cref="CognitiveConfig.GlobalEncodingScale"/>).
    /// </summary>
    GlobalScale = 1
}

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

    /// <summary>
    /// Modo de propagación neuronal (Hybrid por defecto o StrictSpiking).
    /// </summary>
    public NeuromorphicExecutionMode ExecutionMode { get; init; } = NeuromorphicExecutionMode.Hybrid;

    /// <summary>
    /// Modo de codificación TTFS (relativo por muestra o escala global).
    /// </summary>
    public TimeToFirstSpikeMode EncodingMode { get; init; } = TimeToFirstSpikeMode.PerSampleRelative;

    /// <summary>
    /// Escala global utilizada cuando <see cref="EncodingMode"/> es <see cref="TimeToFirstSpikeMode.GlobalScale"/>.
    /// </summary>
    public float GlobalEncodingScale { get; init; } = 1.0f;

    /// <summary>
    /// Habilita depresión sináptica (LTD) en la modulación STDP cuando <c>dt &lt; 0</c>.
    /// </summary>
    public bool EnableStdpDepression { get; init; } = false;
}

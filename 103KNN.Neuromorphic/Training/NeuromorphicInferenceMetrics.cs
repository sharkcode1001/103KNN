namespace _103KNN.Neuromorphic.Training;

/// <summary>
/// Métricas detalladas de una ejecución individual del motor neuromórfico.
/// </summary>
/// <param name="LifFiredCount">Número de neuronas ocultas LIF que superaron su umbral y dispararon.</param>
/// <param name="PropagatedEventCount">Total de eventos propagados entre capas (espigas TTFS de entrada más eventos enviados a capas siguientes, incluidas activaciones sub-umbral).</param>
/// <param name="SynapticOpsCount">Total de operaciones sinápticas ejecutadas por evento al recorrer <c>ConnectedTargets</c> (cuenta operaciones por evento, no conexiones únicas).</param>
public readonly record struct NeuromorphicInferenceMetrics(
    long LifFiredCount,
    long PropagatedEventCount,
    long SynapticOpsCount);

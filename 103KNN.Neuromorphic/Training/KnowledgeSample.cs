namespace _103KNN.Neuromorphic.Training;

/// <summary>
/// Muestra de conocimiento estructurado (par de pregunta/intención y respuesta o hecho conceptual).
/// </summary>
public sealed record KnowledgeSample(
    string Prompt,
    string Completion,
    string Category = "General"
);

/// <summary>
/// Progreso emitido durante el entrenamiento de la red neuromórfica.
/// </summary>
public sealed record TrainingProgressReport(
    int CurrentEpoch,
    int TotalEpochs,
    int SentencesProcessed,
    int VocabSize,
    int SynapsesUpdated,
    double MillisecondsElapsed
);

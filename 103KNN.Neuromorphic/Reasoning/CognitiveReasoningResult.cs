namespace _103KNN.Neuromorphic.Reasoning;

/// <summary>
/// Resultado del proceso de inferencia y deliberación cognitiva neuromórfica.
/// </summary>
public sealed record CognitiveReasoningResult(
    int PredictedIndex,
    float Confidence,
    int DeliberationStepsTaken,
    int TotalSpikesProcessed,
    bool IsEarlyExit,
    IReadOnlyList<float> FinalPotentials,
    TimeSpan ElapsedTime
);

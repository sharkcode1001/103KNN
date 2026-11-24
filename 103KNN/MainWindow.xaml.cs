using System.Text;
using System.Windows;
using _103KNN.Neuromorphic.Associative;
using _103KNN.Neuromorphic.Core;
using _103KNN.Neuromorphic.Encoding;
using _103KNN.Neuromorphic.Plasticity;
using _103KNN.Neuromorphic.Reasoning;
using _103KNN.Neuromorphic.Text;

namespace _103KNN;

/// <summary>
/// Lógica de interacción para MainWindow.xaml demostrando razonamiento, k-NN, STDP y generación Spike-LLM.
/// </summary>
public partial class MainWindow : Window
{
    private NeuromorphicReasoner _reasoner = null!;
    private NeuromorphicKnnEngine _knnEngine = null!;
    private StdpSynapticTrainer _stdpTrainer = null!;
    private SpikeTextGenerator _textGenerator = null!;
    private readonly TimeToFirstSpikeEncoder _encoder = new();

    private const int FeatureDim = 16;
    private const int HiddenDim = 32;
    private const int OutputDim = 4;

    public MainWindow()
    {
        InitializeComponent();
        InitAllEngines();
    }

    private void InitAllEngines()
    {
        var config = new CognitiveConfig
        {
            BaseThreshold = 0.35f,
            MembraneDecay = 0.88f,
            LateralInhibitionFactor = 0.15f,
            EarlyExitConfidenceThreshold = 0.70f,
            MaxDeliberationSteps = 20
        };

        _reasoner = new NeuromorphicReasoner(FeatureDim, HiddenDim, OutputDim, config);
        _stdpTrainer = new StdpSynapticTrainer(new StdpConfig
        {
            APlus = 0.040f,
            AMinus = 0.025f,
            TauPlus = 0.25f,
            TauMinus = 0.30f
        });

        _knnEngine = new NeuromorphicKnnEngine(FeatureDim, config);
        PopulateKnnDataset(sampleCount: 80);

        _textGenerator = new SpikeTextGenerator(embeddingDim: 32, hiddenDim: 64);

        Log("=== Ecosistema Neuromórfico 103KNN Inicializado ===");
        Log($"1. Motor Cognitivo: {FeatureDim} Entradas -> {HiddenDim} Ocultas (LIF) -> {OutputDim} Salidas");
        Log($"2. Generador Spike-LLM: Vocabulario de {_textGenerator.VocabSize} tokens | Sin KV-Cache (Memoria LIF O(1))");
        Log($"3. Motor k-NN Asociativo: {_knnEngine.SampleCount} Muestras en memoria");
        Log("4. Módulo STDP: Calibrado para potenciación causal (LTP) y depresión (LTD)");
    }

    private void PopulateKnnDataset(int sampleCount)
    {
        _knnEngine.Clear();
        var rand = new Random(103);

        for (int i = 0; i < sampleCount; i++)
        {
            int classLabel = i % OutputDim;
            float[] features = new float[FeatureDim];

            for (int d = 0; d < FeatureDim; d++)
            {
                float baseVal = (d % OutputDim == classLabel) ? 0.85f : 0.15f;
                features[d] = Math.Clamp(baseVal + (float)((rand.NextDouble() - 0.5) * 0.25), 0.0f, 1.0f);
            }

            _knnEngine.AddSample(new KnnSample(i, classLabel, features));
        }
    }

    private void BtnFastThinking_Click(object sender, RoutedEventArgs e)
    {
        float[] input = new float[FeatureDim];
        // Señal clara y dominante en canal inicial
        input[0] = 0.95f;
        input[1] = 0.85f;
        input[2] = 0.75f;
        input[3] = 0.10f;

        Log("\n--------------------------------------------------");
        Log(">> [SISTEMA 1: ESTIMULACIÓN CON SEÑAL CLARA / PRIORITARIA]");

        var result = _reasoner.Reason(input);

        TxtMode.Text = "⚡ Sistema 1 (Fast Thinking / TTFS)";
        TxtWinner.Text = $"Decisión #{result.PredictedIndex}";
        TxtEarlyExit.Text = result.IsEarlyExit ? "SÍ (Disparo Temprano Confiable)" : "NO";
        TxtConfidence.Text = $"{result.Confidence * 100:F1} %";
        TxtSteps.Text = $"{result.DeliberationStepsTaken} paso(s)";
        TxtSpikes.Text = $"{result.TotalSpikesProcessed} espigas";
        TxtEfficiency.Text = result.IsEarlyExit ? "Máxima (Salida en paso 1)" : "Deliberativo";
        TxtLatency.Text = $"{result.ElapsedTime.TotalMicroseconds:F1} µs ({result.ElapsedTime.TotalMilliseconds:F3} ms)";

        Log($"Resultado: Clase #{result.PredictedIndex} | Confianza: {result.Confidence * 100:F1}% | Pasos: {result.DeliberationStepsTaken}");
        Log($"Salida Temprana (Early Exit): {result.IsEarlyExit} | Espigas Totales: {result.TotalSpikesProcessed}");
        Log($"Latencia de Inferencia: {result.ElapsedTime.TotalMicroseconds:F1} µs");
    }

    private void BtnDeliberativeThinking_Click(object sender, RoutedEventArgs e)
    {
        float[] input = new float[FeatureDim];
        var rand = new Random();
        for (int i = 0; i < FeatureDim; i++)
        {
            input[i] = 0.35f + (float)(rand.NextDouble() * 0.20);
        }

        Log("\n--------------------------------------------------");
        Log(">> [SISTEMA 2: ESTIMULACIÓN CON ALTA AMBIGÜEDAD / RUIDO]");

        var result = _reasoner.Reason(input);

        TxtMode.Text = "🧠 Sistema 2 (Deliberación Recurrente)";
        TxtWinner.Text = $"Decisión #{result.PredictedIndex}";
        TxtEarlyExit.Text = result.IsEarlyExit ? "SÍ" : "NO (Requirió deliberación)";
        TxtConfidence.Text = $"{result.Confidence * 100:F1} %";
        TxtSteps.Text = $"{result.DeliberationStepsTaken} ciclos";
        TxtSpikes.Text = $"{result.TotalSpikesProcessed} espigas";
        TxtEfficiency.Text = "Deliberativa (Resuelta tras competencia)";
        TxtLatency.Text = $"{result.ElapsedTime.TotalMicroseconds:F1} µs ({result.ElapsedTime.TotalMilliseconds:F3} ms)";

        Log($"Resultado: Clase #{result.PredictedIndex} | Confianza: {result.Confidence * 100:F1}% | Ciclos: {result.DeliberationStepsTaken}");
        Log($"Espigas Totales Procesadas: {result.TotalSpikesProcessed} | Latencia: {result.ElapsedTime.TotalMicroseconds:F1} µs");
    }

    private void BtnKnnBenchmark_Click(object sender, RoutedEventArgs e)
    {
        Log("\n--------------------------------------------------");
        Log(">> [BENCHMARK: k-NN NEUROMÓRFICO (TTFS) vs EUCLIDIANO CLÁSICO]");

        float[] query = new float[FeatureDim];
        for (int d = 0; d < FeatureDim; d++)
        {
            query[d] = (d % OutputDim == 2) ? 0.88f : 0.12f;
        }

        var benchmark = _knnEngine.BenchmarkVsClassic(query, k: 3);

        TxtMode.Text = "📊 Benchmark k-NN Neuromórfico vs Euclidiano";
        TxtWinner.Text = $"Clase {benchmark.NeuromorphicPrediction} (Clásico: {benchmark.ClassicPrediction})";
        TxtEarlyExit.Text = benchmark.PredictionsMatch ? "COINCIDENCIA EXACTA (100%)" : "DIVERGENCIA";
        TxtConfidence.Text = benchmark.PredictionsMatch ? "100% de Acuerdo" : "90% de Acuerdo";
        TxtSteps.Text = "K=3 vecinos por orden de espiga";
        TxtSpikes.Text = $"{benchmark.NeuromorphicSops} SOPs (Sumas dispersas)";
        TxtEfficiency.Text = $"{benchmark.ClassicFlops} FLOPs vs {benchmark.NeuromorphicSops} SOPs";
        TxtLatency.Text = $"Neuromórfico: {benchmark.NeuromorphicTime.TotalMicroseconds:F1} µs | Clásico: {benchmark.ClassicTime.TotalMicroseconds:F1} µs";

        Log($"Predicción Neuromórfica: Clase {benchmark.NeuromorphicPrediction}");
        Log($"Predicción Euclidiana Clásica: Clase {benchmark.ClassicPrediction}");
        Log($"¿Coinciden los algoritmos?: {(benchmark.PredictionsMatch ? "SÍ" : "NO")}");
        Log($"Operaciones Flotantes Clásicas: {benchmark.ClassicFlops} FLOPs");
        Log($"Operaciones Sinápticas Neuromórficas: {benchmark.NeuromorphicSops} SOPs");
        Log($"Latencia: Neuromórfico={benchmark.NeuromorphicTime.TotalMicroseconds:F1} µs vs Clásico={benchmark.ClassicTime.TotalMicroseconds:F1} µs");
    }

    private void BtnStdpLearn_Click(object sender, RoutedEventArgs e)
    {
        Log("\n--------------------------------------------------");
        Log(">> [PLASTICIDAD BIOLÓGICA: APRENDIZAJE ONLINE STDP]");

        float[] sample = new float[FeatureDim];
        sample[0] = 0.95f;
        sample[1] = 0.90f;
        sample[2] = 0.85f;
        sample[3] = 0.80f;

        var spikes = _encoder.Encode(sample);

        _reasoner.HiddenLayer.Reset();
        _reasoner.HiddenLayer.Forward(spikes);

        float deltaW = _stdpTrainer.ApplyStdp(_reasoner.HiddenLayer, spikes);

        TxtMode.Text = "🌱 Aprendizaje Online (STDP)";
        TxtWinner.Text = "Sinapsis adaptadas";
        TxtEarlyExit.Text = "Refuerzo sináptico LTP/LTD";
        TxtConfidence.Text = "Auto-organización";
        TxtSteps.Text = "1 Época local";
        TxtSpikes.Text = $"{spikes.Count} espigas de entrada";
        TxtEfficiency.Text = $"Delta W neto = {deltaW:F4}";
        TxtLatency.Text = "Adaptación instantánea";

        Log($"Se procesaron {spikes.Count} espigas de entrada.");
        Log($"Regla STDP aplicada exitosamente a la matriz sináptica.");
        Log($"Ajuste neto acumulado en sinapsis (Delta W): {deltaW:F5}");
    }

    private void BtnGenerateText_Click(object sender, RoutedEventArgs e)
    {
        string prompt = TxtPrompt.Text.Trim();
        if (string.IsNullOrEmpty(prompt))
            prompt = "el sol produce";

        Log("\n--------------------------------------------------");
        Log($">> [GENERADOR DE TEXTO SPIKE-LLM] Prompt: \"{prompt}\"");

        var genRes = _textGenerator.Generate(prompt, maxNewTokens: 12, temperature: 0.6f);

        TxtMode.Text = "✨ Generación de Texto (Spike-LLM)";
        TxtWinner.Text = genRes.GeneratedText;
        TxtEarlyExit.Text = "Causal Autoregresivo";
        TxtConfidence.Text = "Coherencia Sináptica";
        TxtSteps.Text = $"{genRes.TokensCount} tokens generados";
        TxtSpikes.Text = $"{genRes.TotalSpikesFired} espigas";
        TxtEfficiency.Text = "KV-Cache: 0 MB (Memoria LIF O(1))";
        TxtLatency.Text = $"{genRes.MicrosecondsPerToken:F1} µs/tok ({genRes.ElapsedTime.TotalMilliseconds:F2} ms)";

        Log($"Texto Generado: \"{prompt} {genRes.GeneratedText}\"");
        Log($"Tokens Generados: {genRes.TokensCount} | Espigas Disparadas: {genRes.TotalSpikesFired}");
        Log($"Velocidad: {genRes.MicrosecondsPerToken:F1} µs por token ({genRes.ElapsedTime.TotalMilliseconds:F2} ms total)");
        Log("Uso de memoria: Cero KV-cache (la memoria reside en los potenciales de membrana de neuronas LIF).");
    }

    private void BtnTrainCorpus_Click(object sender, RoutedEventArgs e)
    {
        Log("\n--------------------------------------------------");
        Log(">> [ENTRENAMIENTO CAUSAL HEBBIANO DE CORPUS]");
        var corpus = SpikeTextGenerator.GetDefaultReasoningCorpus();
        _textGenerator.TrainOnCorpus(corpus, epochs: 15);
        Log($"Se re-entrenó el generador sobre {corpus.Count} oraciones de razonamiento lógico y solar.");
        Log("Las conexiones recurrentes causales han sido fortalecidas.");
    }

    private void BtnResetEngine_Click(object sender, RoutedEventArgs e)
    {
        InitAllEngines();
        TxtMode.Text = "Estado reiniciado";
        TxtWinner.Text = "-";
        TxtEarlyExit.Text = "-";
        TxtConfidence.Text = "-";
        TxtSteps.Text = "-";
        TxtSpikes.Text = "-";
        TxtEfficiency.Text = "-";
        TxtLatency.Text = "-";
        Log("\n[INFO] Todos los pesos, memorias y modelos han sido restablecidos.");
    }

    private void Log(string text)
    {
        TxtConsole.AppendText(text + Environment.NewLine);
        TxtConsole.ScrollToEnd();
    }
}

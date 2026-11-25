using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
using _103KNN.Neuromorphic.Text;

namespace _103KNN.Neuromorphic.Training;

/// <summary>
/// Entrenador de conocimiento nativo para redes neuromórficas.
/// Permite entrenar el modelo con corpus temáticos predefinidos (C#, SNN, Algoritmos, Diálogo)
/// o ingerir archivos externos de texto (.txt, .md, .json) para expandir el conocimiento de la red.
/// </summary>
public sealed class NeuromorphicKnowledgeTrainer
{
    /// <summary>
    /// Retorna una rica base de conocimientos nativa estructurada en pares de diálogo y hechos causales.
    /// </summary>
    public static List<string> GetComprehensiveNativeCorpus()
    {
        return new List<string>
        {
            // === Saludos, Identidad y Cortesía ===
            "usuario: hola asistente: hola un gusto saludarte en que puedo colaborar hoy",
            "usuario: hola que tal asistente: hola que tal estoy lista para ayudarte a razonar y responder preguntas tecnicas",
            "usuario: hola buen dia asistente: muy buen dia como estas estoy a tu disposicion para explorar informacion y programar",
            "usuario: como te llamas asistente: me llamo 103knn bionet soy una red neuronal neuromorfica bioinspirada",
            "usuario: quien eres asistente: soy un modelo neuromorfico que razona mediante impulsos de espigas lif y memoria asociativa",
            "usuario: que puedes hacer asistente: puedo explicar conceptos de programacion razonar sobre redes neuromorficas clasificar datos con knn y dialogar",
            "usuario: como funcionas asistente: funciono mediante dinamica de espigas temporales ttfs plasticidad sinaptica stdp y memoria recurrente sin kv-cache",
            "usuario: gracias asistente: no hay de que siempre es un placer ayudarte",
            "usuario: muchas gracias asistente: con mucho gusto aqui estare si necesitas mas asistencia",
            "usuario: adios asistente: hasta pronto que tengas un excelente dia",

            // === C# y Plataforma .NET 10 ===
            "usuario: que sabes de c# asistente: c# es un lenguaje de programacion moderno orientado a objetos fuertemente tipado desarrollado por microsoft",
            "usuario: que es c# asistente: c# es un lenguaje de programacion moderno y potente diseñado para crear aplicaciones robustas en la plataforma dotnet",
            "usuario: conoces del lenguaje c# asistente: si conozco c# a fondo incluye clases interfaces linq tareas asincronas records y alto rendimiento",
            "usuario: que caracteristicas tiene c# asistente: c# cuenta con tipado estatico recoleccion de basura pattern matching linq async await y rendimiento nativo aot",
            "usuario: que es dotnet 10 asistente: dotnet 10 es el framework unificado multiplataforma y de codigo abierto de microsoft con optimizaciones de vanguardia",
            "usuario: como se declara una clase en c# asistente: se declara usando la palabra clave class junto con sus propiedades constructores y metodos publicos o privados",
            "usuario: que es linq en c# asistente: linq o language integrated query permite consultar colecciones objetos y bases de datos con sintaxis declarativa limpia",
            "usuario: que es async y await en c# asistente: async y await permiten programacion asincrona sin bloquear el hilo de ejecucion facilitando alta concurrencia",
            "usuario: que son los records en c# asistente: los records son tipos de datos inmutables con igualdad basada en valores y sintaxis concisa introducidos en versiones modernas",
            "c# es un lenguaje de programacion potente desarrollado por microsoft para la plataforma dotnet",
            "en c# el recolector de basura gestiona la memoria de forma automatica y eficiente",
            "el compilador de c# genera codigo intermedio cil que se ejecuta sobre el runtime clr de dotnet",
            "las interfaces en c# definen contratos que las clases e implementaciones concretas deben cumplir",
            "las expresiones lambda y linq facilitan el filtrado y transformacion funcional de datos en colecciones",
            "la palabra clave async en c# marca metodos asincronos que retornan task y permiten esperas con await",

            // === Redes Neuronales Neuromórficas y Bio-Inspiración ===
            "usuario: que es una red neuromorfica asistente: es una red inspirada en el cerebro biologico que procesa espigas temporales en lugar de operaciones matriciales continuas",
            "usuario: como razona la red neuromorfica asistente: razona integrando potenciales electricos en neuronas lif y decidiendo mediante el tiempo de primer disparo ttfs",
            "usuario: que es una neurona lif asistente: lif significa leaky integrate and fire acumula potencial de membrana con fuga y dispara al superar el umbral",
            "usuario: que es ttfs asistente: ttfs significa time to first spike donde las caracteristicas con mayor activacion o prioridad disparan primero",
            "usuario: por que no usas kv-cache asistente: no uso kv-cache porque el estado contextual se retiene en los potenciales de membrana de las neuronas recurrentes en memoria constante o(1)",
            "usuario: que es la plasticidad stdp asistente: stdp es spike timing dependent plasticity una regla biologica que ajusta pesos segun el orden temporal causal de los disparos",
            "las neuronas lif integran corrientes sinapticas en su membrana y emiten espigas binarias discretas",
            "la inhibicion lateral entre neuronas promueve la competencia y suprime respuestas ruidosas",
            "el modelo biologico procesa eventos dispersos de forma orientada a eventos ahorrando energia",
            "la memoria asociativa almacena secuencias y representaciones conceptuales en la conectividad sinaptica",

            // === Algoritmo k-NN y Razonamiento Asociativo ===
            "usuario: que es knn asistente: knn o k nearest neighbors es un metodo que clasifica un punto segun la clase predominante de sus vecinos mas cercanos",
            "usuario: que ventajas tiene 103knn asistente: 103knn encuentra vecinos mediante latencia temporal de espigas ttfs logrando complejidad o(activas) sin distancias euclidianas densas",
            "usuario: como clasifica knn asistente: compara la muestra consultada con las muestras almacenadas en el espacio vectorial y selecciona los k ejemplares mas proximos",
            "el clasificador knn no requiere fase de optimizacion por gradiente sino memoria asociativa directa",
            "el motor 103knn utiliza neuronas lif como detectores de coincidencia temporal para ordenar vecinos",

            // === Energía Solar, Fotovoltaica y Física ===
            "usuario: que es la energia solar asistente: es la radiacion electromagnetica emitida por el sol capturada por paneles fotovoltaicos para generar electricidad limpia",
            "usuario: como funciona un panel fotovoltaico asistente: aprovecha el efecto fotoelectrico en celdas de silicio para convertir fotones de luz en corriente continua",
            "usuario: que es la irradiancia solar asistente: es la potencia incidente de radiacion solar por unidad de superficie medida normalmente en vatios por metro cuadrado",
            "el sol emite radiacion solar limpia indispensable para la vida y la generacion de energia renovable",
            "los inversores solares transforman la corriente continua de los paneles en corriente alterna para la red electrica",
            "la orientacion e inclinacion de los modulos fotovoltaicos maximiza el aprovechamiento de la luz solar"
        };
    }

    /// <summary>
    /// Ingiere un archivo de texto (.txt, .md o .json) y extrae oraciones de entrenamiento válidas,
    /// soportando formatos mono-línea y multi-línea (usuario: ... \n asistente: ...).
    /// </summary>
    public static List<string> ParseTextFile(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"El archivo no existe: {filePath}");

        string content = File.ReadAllText(filePath);
        var sentences = new List<string>();

        // Si es JSON, intentar parsear pares prompt/completion o lista de strings
        if (filePath.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            try
            {
                using var doc = JsonDocument.Parse(content);
                if (doc.RootElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var item in doc.RootElement.EnumerateArray())
                    {
                        if (item.ValueKind == JsonValueKind.String)
                        {
                            string? s = item.GetString();
                            if (!string.IsNullOrWhiteSpace(s)) sentences.Add(s);
                        }
                        else if (item.ValueKind == JsonValueKind.Object)
                        {
                            string p = item.TryGetProperty("prompt", out var pv) ? pv.GetString() ?? "" : "";
                            string c = item.TryGetProperty("completion", out var cv) ? cv.GetString() ?? "" : "";
                            if (!string.IsNullOrWhiteSpace(p) && !string.IsNullOrWhiteSpace(c))
                            {
                                sentences.Add($"usuario: {CleanText(p)} asistente: {CleanText(c)}");
                            }
                        }
                    }
                }
            }
            catch
            {
                // Fallback a parseo de texto plano si el JSON no tiene ese esquema
            }
        }

        if (sentences.Count == 0)
        {
            var rawLines = content.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries);
            string? pendingUserQuery = null;

            foreach (var rawLine in rawLines)
            {
                string line = rawLine.Trim();
                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#") || line.StartsWith("//"))
                    continue;

                // 1. Detectar si la misma línea contiene pregunta y respuesta juntas (P: ... R: ...)
                var matchInline = Regex.Match(line, @"^(?:p|pregunta|q|usuario):\s*(.+?)\s*(?:r|respuesta|a|asistente):\s*(.+)$", RegexOptions.IgnoreCase);
                if (matchInline.Success)
                {
                    string q = CleanText(matchInline.Groups[1].Value);
                    string a = CleanText(matchInline.Groups[2].Value);
                    sentences.Add($"usuario: {q} asistente: {a}");
                    pendingUserQuery = null;
                    continue;
                }

                // 2. Detectar inicio de pregunta en formato multi-línea (usuario: ...)
                var matchUser = Regex.Match(line, @"^(?:p|pregunta|q|usuario):\s*(.+)$", RegexOptions.IgnoreCase);
                if (matchUser.Success)
                {
                    pendingUserQuery = CleanText(matchUser.Groups[1].Value);
                    continue;
                }

                // 3. Detectar respuesta que complementa la pregunta previa multi-línea (asistente: ...)
                var matchAssistant = Regex.Match(line, @"^(?:r|respuesta|a|asistente):\s*(.+)$", RegexOptions.IgnoreCase);
                if (matchAssistant.Success && pendingUserQuery != null)
                {
                    string a = CleanText(matchAssistant.Groups[1].Value);
                    sentences.Add($"usuario: {pendingUserQuery} asistente: {a}");
                    pendingUserQuery = null;
                    continue;
                }

                // 4. Si era una línea regular de texto/hecho factual
                var parts = Regex.Split(line, @"(?<=[.!?])\s+");
                foreach (var part in parts)
                {
                    string cleaned = CleanText(part);
                    if (cleaned.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length >= 3)
                    {
                        sentences.Add(cleaned);
                    }
                }
            }
        }

        return sentences;
    }

    /// <summary>
    /// Limpia y normaliza texto para entrenamiento en la SNN.
    /// </summary>
    public static string CleanText(string input)
    {
        if (string.IsNullOrWhiteSpace(input)) return string.Empty;

        // Normalizar acentos y caracteres especiales si se desea, o mantenerlos legibles
        string s = input.ToLowerInvariant();
        s = s.Replace("á", "a").Replace("é", "e").Replace("í", "i").Replace("ó", "o").Replace("ú", "u").Replace("ñ", "n");
        s = Regex.Replace(s, @"[^\w\s\:\?\!\.\,\-]", " ");
        s = Regex.Replace(s, @"\s+", " ").Trim();
        return s;
    }

    /// <summary>
    /// Entrena de manera asíncrona un SpikeTextGenerator con un conjunto de oraciones,
    /// reportando el progreso a la UI.
    /// </summary>
    public static async Task<TrainingProgressReport> TrainAsync(
        SpikeTextGenerator generator,
        IEnumerable<string> sentences,
        int epochs = 15,
        float learningRate = 0.22f,
        IProgress<TrainingProgressReport>? progress = null)
    {
        var sentenceList = sentences.ToList();
        var sw = Stopwatch.StartNew();

        int totalSynapses = 0;

        await Task.Run(() =>
        {
            // 1. Expandir vocabulario
            generator.Tokenizer.BuildVocabulary(sentenceList);

            // 2. Entrenar capa recurrente, asociaciones semánticas y transiciones causales
            totalSynapses = generator.TrainOnCorpusDetailed(sentenceList, epochs, learningRate, (ep, syn) =>
            {
                progress?.Report(new TrainingProgressReport(
                    CurrentEpoch: ep,
                    TotalEpochs: epochs,
                    SentencesProcessed: sentenceList.Count,
                    VocabSize: generator.VocabSize,
                    SynapsesUpdated: syn,
                    MillisecondsElapsed: sw.Elapsed.TotalMilliseconds
                ));
            });
        });

        sw.Stop();

        var finalReport = new TrainingProgressReport(
            CurrentEpoch: epochs,
            TotalEpochs: epochs,
            SentencesProcessed: sentenceList.Count,
            VocabSize: generator.VocabSize,
            SynapsesUpdated: totalSynapses,
            MillisecondsElapsed: sw.Elapsed.TotalMilliseconds
        );

        progress?.Report(finalReport);
        return finalReport;
    }
}

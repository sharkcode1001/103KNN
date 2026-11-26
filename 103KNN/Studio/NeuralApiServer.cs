using System;
using System.IO;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace _103KNN.Studio;

public class NeuralApiServer
{
    private readonly HttpListener _listener;
    private readonly MainWindow _mainWindow;
    private bool _isRunning;
    private static readonly JsonSerializerOptions JsonOpts = new() { PropertyNameCaseInsensitive = true, WriteIndented = true };

    public NeuralApiServer(MainWindow mainWindow, int port = 5000)
    {
        _mainWindow = mainWindow;
        _listener = new HttpListener();
        _listener.Prefixes.Add($"http://localhost:{port}/");
    }

    public bool Start()
    {
        try
        {
            _isRunning = true;
            _listener.Start();
            Task.Run(HandleIncomingConnections);
            return true;
        }
        catch
        {
            _isRunning = false;
            return false;
        }
    }

    public void Stop()
    {
        _isRunning = false;
        try { _listener.Stop(); } catch { }
    }

    private async Task HandleIncomingConnections()
    {
        while (_isRunning)
        {
            try
            {
                var context = await _listener.GetContextAsync();
                _ = ProcessRequest(context);
            }
            catch { }
        }
    }

    private async Task ProcessRequest(HttpListenerContext context)
    {
        string path = context.Request.Url?.AbsolutePath.ToLowerInvariant() ?? "";
        string body = "";
        if (context.Request.HasEntityBody)
        {
            using var reader = new StreamReader(context.Request.InputStream, context.Request.ContentEncoding);
            body = await reader.ReadToEndAsync();
        }

        object responseData = new { status = "error", message = "Endpoint not found. GET /help for available commands." };

        try
        {
            switch (path)
            {
                case "/":
                case "/help":
                    responseData = new
                    {
                        status = "ok",
                        endpoints = new[]
                        {
                            "GET  /status",
                            "POST /ui/view                {\"view\":\"designer|trainer|benchmark\"}",
                            "POST /design                 {\"NetworkType\":3,\"Inputs\":8,\"HiddenLayers\":[32,8],\"Outputs\":2}",
                            "POST /build",
                            "POST /dataset/load-csv       {\"path\":\"E:\\\\GitHub\\\\103KNN\\\\datasets\\\\8 in 2 out.csv\"}",
                            "POST /dataset/generate-synthetic {\"sampleCount\":240}",
                            "POST /benchmark/run          {\"architectures\":[0,1,2,3,4,5],\"maxEpochs\":2000,\"targetError\":0.0001,\"learningRate\":0.005,\"batchSize\":64}",
                            "GET  /benchmark/results",
                            "POST /train/start",
                            "POST /train/load-data        {\"Inputs\":[[...]],\"Targets\":[[...]]}",
                            "POST /train/execute",
                            "GET  /train/progress",
                            "POST /train/stop",
                            "POST /predict                [0.5, 0.5]"
                        },
                        architectures = new
                        {
                            Nowral = 0, LibQuid = 1, NewNow = 2, Neuromorphic103KNN = 3,
                            FastTree = 4, FastForest = 5, StandardMlp = 6, ResNetTabular = 7, Lstm = 8
                        }
                    };
                    break;

                case "/status":
                    responseData = _mainWindow.Dispatcher.Invoke(() => _mainWindow.GetRemoteStatus());
                    break;

                case "/ui/view":
                    var viewReq = JsonSerializer.Deserialize<ViewRequest>(body, JsonOpts);
                    if (viewReq != null)
                    {
                        _mainWindow.Dispatcher.Invoke(() => _mainWindow.SwitchViewByName(viewReq.View));
                        responseData = new { status = "ok", view = viewReq.View };
                    }
                    break;

                case "/design":
                    var design = JsonSerializer.Deserialize<DesignRequest>(body, JsonOpts);
                    if (design != null)
                    {
                        _mainWindow.Dispatcher.Invoke(() =>
                            _mainWindow.ApplyRemoteDesign(design.NetworkType, design.Inputs, design.HiddenLayers, design.Outputs));
                        responseData = new { status = "ok", message = "Structure applied" };
                    }
                    break;

                case "/build":
                    await _mainWindow.Dispatcher.InvokeAsync(async () => await _mainWindow.BuildNetworkWithAnimation());
                    responseData = new { status = "ok", message = "Build completed" };
                    break;

                case "/dataset/load-csv":
                    var csvReq = JsonSerializer.Deserialize<LoadCsvRequest>(body, JsonOpts);
                    if (csvReq != null && !string.IsNullOrWhiteSpace(csvReq.Path))
                    {
                        responseData = _mainWindow.Dispatcher.Invoke(() => _mainWindow.LoadRemoteCsv(csvReq.Path));
                    }
                    break;

                case "/dataset/generate-synthetic":
                    var synReq = string.IsNullOrWhiteSpace(body)
                        ? new SyntheticRequest()
                        : JsonSerializer.Deserialize<SyntheticRequest>(body, JsonOpts) ?? new SyntheticRequest();
                    responseData = _mainWindow.Dispatcher.Invoke(() => _mainWindow.GenerateRemoteSynthetic(synReq.SampleCount));
                    break;

                case "/benchmark/run":
                    var benchReq = string.IsNullOrWhiteSpace(body)
                        ? new BenchmarkRunRequest()
                        : JsonSerializer.Deserialize<BenchmarkRunRequest>(body, JsonOpts) ?? new BenchmarkRunRequest();
                    responseData = await _mainWindow.RunRemoteBenchmarkAsync(benchReq);
                    break;

                case "/benchmark/results":
                    responseData = _mainWindow.Dispatcher.Invoke(() => _mainWindow.GetRemoteBenchmarkResults());
                    break;

                case "/train/start":
                    _mainWindow.Dispatcher.Invoke(() => _mainWindow.OpenTrainerView());
                    responseData = new { status = "ok", message = "Trainer opened" };
                    break;

                case "/train/load-data":
                    var data = JsonSerializer.Deserialize<TrainingDataRequest>(body, JsonOpts);
                    if (data != null)
                    {
                        _mainWindow.Dispatcher.Invoke(() => _mainWindow.SetRemoteData(data.Inputs, data.Targets));
                        responseData = new { status = "ok", message = "Data loaded" };
                    }
                    break;

                case "/train/execute":
                    _mainWindow.Dispatcher.Invoke(() => _mainWindow.ExecuteRemoteTraining());
                    responseData = new { status = "ok", message = "Training started" };
                    break;

                case "/train/progress":
                    responseData = _mainWindow.Dispatcher.Invoke(() => _mainWindow.GetTrainingProgress());
                    break;

                case "/train/stop":
                    _mainWindow.Dispatcher.Invoke(() => _mainWindow.StopRemoteTraining());
                    responseData = new { status = "ok", message = "Training stopped" };
                    break;

                case "/predict":
                    var input = JsonSerializer.Deserialize<float[]>(body, JsonOpts);
                    if (input != null)
                        responseData = _mainWindow.Dispatcher.Invoke(() => _mainWindow.ExecuteRemotePredict(input));
                    break;
            }
        }
        catch (Exception ex)
        {
            responseData = new { status = "error", message = ex.Message };
        }

        byte[] buffer = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(responseData, JsonOpts));
        context.Response.ContentType = "application/json";
        await context.Response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
        context.Response.Close();
    }
}

public class ViewRequest
{
    public string View { get; set; } = "benchmark";
}

public class LoadCsvRequest
{
    public string Path { get; set; } = "";
}

public class SyntheticRequest
{
    public int SampleCount { get; set; } = 240;
}

public class BenchmarkRunRequest
{
    public int[] Architectures { get; set; } = new[] { 0, 1, 2, 3, 4, 5 };
    public long MaxEpochs { get; set; } = 2000;
    public float TargetError { get; set; } = 0.0001f;
    public double? LearningRate { get; set; }
    public int? BatchSize { get; set; }
    public bool TrainTestSplit { get; set; } = true;
}

public class DesignRequest
{
    public int NetworkType { get; set; }
    public int[] HiddenLayers { get; set; } = Array.Empty<int>();
    public int Inputs { get; set; }
    public int Outputs { get; set; }
}

public class TrainingDataRequest
{
    public float[][] Inputs { get; set; } = Array.Empty<float[]>();
    public float[][] Targets { get; set; } = Array.Empty<float[]>();
}

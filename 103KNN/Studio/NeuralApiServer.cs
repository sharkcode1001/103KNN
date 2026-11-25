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

        object responseData = new { status = "error", message = "Endpoint no encontrado" };

        try
        {
            switch (path)
            {
                case "/status":
                    responseData = _mainWindow.Dispatcher.Invoke(() => _mainWindow.GetRemoteStatus());
                    break;

                case "/design":
                    var design = JsonSerializer.Deserialize<DesignRequest>(body);
                    if (design != null)
                    {
                        _mainWindow.Dispatcher.Invoke(() =>
                            _mainWindow.ApplyRemoteDesign(design.NetworkType, design.Inputs, design.HiddenLayers, design.Outputs));
                        responseData = new { status = "ok", message = "Estructura aplicada" };
                    }
                    break;

                case "/build":
                    await _mainWindow.Dispatcher.InvokeAsync(async () => await _mainWindow.BuildNetworkWithAnimation());
                    responseData = new { status = "ok", message = "Construcción iniciada" };
                    break;

                case "/train/start":
                    _mainWindow.Dispatcher.Invoke(() => _mainWindow.OpenTrainerView());
                    responseData = new { status = "ok", message = "Entrenador abierto" };
                    break;

                case "/train/load-data":
                    var data = JsonSerializer.Deserialize<TrainingDataRequest>(body);
                    if (data != null)
                    {
                        _mainWindow.Dispatcher.Invoke(() => _mainWindow.SetRemoteData(data.Inputs, data.Targets));
                        responseData = new { status = "ok", message = "Datos cargados correctamente" };
                    }
                    break;

                case "/train/execute":
                    _mainWindow.Dispatcher.Invoke(() => _mainWindow.ExecuteRemoteTraining());
                    responseData = new { status = "ok", message = "Ciclo de entrenamiento iniciado" };
                    break;

                case "/train/progress":
                    responseData = _mainWindow.Dispatcher.Invoke(() => _mainWindow.GetTrainingProgress());
                    break;

                case "/train/stop":
                    _mainWindow.Dispatcher.Invoke(() => _mainWindow.StopRemoteTraining());
                    responseData = new { status = "ok", message = "Entrenamiento detenido" };
                    break;

                case "/predict":
                    var input = JsonSerializer.Deserialize<float[]>(body);
                    if (input != null)
                        responseData = _mainWindow.Dispatcher.Invoke(() => _mainWindow.ExecuteRemotePredict(input));
                    break;
            }
        }
        catch (Exception ex)
        {
            responseData = new { status = "error", message = ex.Message };
        }

        byte[] buffer = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(responseData));
        context.Response.ContentType = "application/json";
        await context.Response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
        context.Response.Close();
    }
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

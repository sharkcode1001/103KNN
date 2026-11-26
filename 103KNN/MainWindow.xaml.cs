using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Threading;
using Microsoft.Win32;
using _103KNN.Studio;
using NowLib;

namespace _103KNN;

public partial class MainWindow : Window
{
    private readonly GlobalNetworkSettings _networkSettings = new();
    private readonly TrainingSettings _trainingSettings = new();
    private readonly ModelTrainerEngine _trainerEngine = new();
    private readonly ObservableCollection<BenchmarkResultItem> _benchmarkResults = new();
    private NeuralApiServer? _httpServer;
    private CancellationTokenSource? _benchCts;
    private bool _isDarkTheme = false;
    private bool _isEnglish = true;
    private DispatcherTimer? _structureDebounce;

    public MainWindow()
    {
        InitializeComponent();
        BenchDataGrid.ItemsSource = _benchmarkResults;
        InitStudio();
        ApplyTheme(_isDarkTheme);
        ApplyLanguage(_isEnglish);
    }

    private void InitStudio()
    {
        var layers = _networkSettings.ParseStructure();
        if (layers.Count >= 2)
        {
            DesignerCanvas.UpdateStructure(layers);
        }

        DesignerCanvas.StructureChanged += (_, _) =>
        {
            UpdateTopologyStats();
            if (!_trainerEngine.IsTraining) EnsureMapsInitialized();
        };
        UpdateTopologyStats();

        _trainerEngine.OnLog += msg => Dispatcher.Invoke(() => TrainerLog(msg));
        _trainerEngine.OnProgressUpdated += () => Dispatcher.Invoke(() =>
        {
            UpdateTrainerStatusUI();
            if (_trainerEngine.CurrentEpoch > 0)
            {
                LossChart.AddPoint(_trainerEngine.CurrentEpoch, _trainerEngine.CurrentMSE);
            }
        });
        _trainerEngine.OnTrainingStateChanged += training => Dispatcher.Invoke(() =>
        {
            BtnTrainerStart.IsEnabled = !training;
            BtnTrainerStop.IsEnabled = training;
        });
        _trainerEngine.OnTopologyChanged += maps => Dispatcher.Invoke(() =>
        {
            TrainerCanvas.LoadExistingStructure(maps);
            DesignerCanvas.LoadExistingStructure(maps);
            UpdateTopologyStats();
        });

        _httpServer = new NeuralApiServer(this, 5000);
        bool started = _httpServer.Start();
        TxtServerStatus.Text = started
            ? (_isEnglish ? "API: localhost:5000 (Active)" : "API: localhost:5000 (Activa)")
            : (_isEnglish ? "API: port 5000 busy" : "API: puerto 5000 ocupado");

        TxtActiveModelBadge.Text = FormatArchName(SelectedArchitecture);
        DesignerLog(_isEnglish ? "103KNN Neural Studio initialized." : "103KNN Neural Studio inicializado.");
    }

    protected override void OnClosed(EventArgs e)
    {
        _benchCts?.Cancel();
        _httpServer?.Stop();
        _trainerEngine.Dispose();
        base.OnClosed(e);
    }

    // --- Cambio de Idioma (EN / ES) ---

    private void BtnLangToggle_Click(object sender, RoutedEventArgs e)
    {
        _isEnglish = !_isEnglish;
        ApplyLanguage(_isEnglish);
    }

    private void ApplyLanguage(bool en)
    {
        BtnLangToggle.Content = en ? "🌐 ES" : "🌐 EN";
        BtnThemeToggle.Content = _isDarkTheme
            ? (en ? "☀️ Light Theme" : "☀️ Tema Claro")
            : (en ? "🌙 Dark Theme" : "🌙 Tema Oscuro");

        TxtHeaderSubtitle.Text = en
            ? "Interactive Topology Design • Multi-Architecture Training • Efficiency Benchmark"
            : "Diseño Interactivo de Topologías • Entrenamiento Multi-Arquitectura • Comparativa de Eficiencia";
        LblHeaderArchPrefix.Text = en ? "Architecture: " : "Arquitectura: ";

        NavDesigner.Content = en ? "Architecture Designer" : "Diseñador de Arquitectura";
        NavTrainer.Content = en ? "Trainer & Evaluation" : "Entrenador y Evaluación";
        NavBenchmark.Content = en ? "Architecture Comparator" : "Comparador de Arquitecturas";

        // Diseñador
        LblTopologyTitle.Text = en ? "Network Topology" : "Topología de Red";
        LblTopologyHint.Text = en
            ? "Click 2 adjacent neurons to toggle connections"
            : "Clic en 2 neuronas adyacentes para editar conexiones";
        LblDesignerLogTitle.Text = en ? "DESIGN LOG" : "REGISTRO DE DISEÑO";
        LblArchTypeTitle.Text = en ? "ARCHITECTURE TYPE" : "TIPO DE ARQUITECTURA";

        CmbArchNowral.Content = en ? "Nowral (Static)" : "Nowral (Estática)";
        CmbArchLibQuid.Content = en ? "LibQuid (Liquid)" : "LibQuid (Líquida)";
        CmbArchNewNow.Content = en ? "NewNow (Dynamic)" : "NewNow (Dinámica)";
        CmbArchNeuro.Content = "103KNN Neuromorphic";
        CmbArchFastTree.Content = en ? "FastTree (GBDT Tabular)" : "FastTree (GBDT Tabular)";
        CmbArchFastForest.Content = "FastForest (Random Forest)";
        CmbArchStdMlp.Content = "Standard MLP (PyTorch)";
        CmbArchResNet.Content = "ResNet-Tabular (Skip)";
        CmbArchLstm.Content = en ? "LSTM (Recurrent)" : "LSTM (Recurrente)";

        BtnShowNetProps.Content = en ? "Network Props" : "Propiedades Red";
        BtnShowTrainProps.Content = en ? "Training Config" : "Config. Entreno";
        LblStructure.Text = en ? "Structure (neurons per layer — auto-applies)" : "Estructura (neuronas por capa — auto-aplica)";
        LblActivation.Text = en ? "Activation Function" : "Función de Activación";
        LblOptimizer.Text = en ? "Optimizer" : "Optimizador";
        LblStdpRate.Text = en ? "STDP Rate (103KNN Neuromorphic)" : "Tasa STDP (103KNN Neuromorphic)";

        BtnFullConnect.Content = en ? "Restore Connections" : "Restaurar Conexiones";
        BtnStartTraining.Content = en ? "Build & Train →" : "Construir y Entrenar →";

        // Entrenador
        LblKpiDataset.Text = "DATASET";
        LblKpiEpoch.Text = en ? "CURRENT EPOCH" : "ÉPOCA ACTUAL";
        LblKpiMse.Text = en ? "MSE LOSS" : "ERROR MSE";
        LblTrainerOpsTitle.Text = SelectedArchitecture == NetworkArchitectureType.Neuromorphic
            ? (en ? "SPIKES / SAMPLE" : "ESPIGAS / MUESTRA")
            : (en ? "OPS / INFERENCE" : "OPS / INFERENCIA");

        LblLossChartTitle.Text = en ? "CONVERGENCE CURVE (REAL-TIME MSE)" : "CURVA DE CONVERGENCIA (MSE EN TIEMPO REAL)";
        LblTrainerLogTitle.Text = en ? "TRAINING CONSOLE" : "CONSOLA DE ENTRENAMIENTO";
        BtnSaveModel.Content = en ? "Save Model" : "Guardar Modelo";
        BtnLoadModel.Content = en ? "Load Model" : "Cargar Modelo";
        LblDatasetGoalTitle.Text = en ? "DATASET & TARGET" : "DATASET Y OBJETIVO";
        BtnLoadCsv.Content = en ? "Load CSV" : "Cargar CSV";
        BtnLoadParquet.Content = en ? "Load Parquet" : "Cargar Parquet";
        LblTargetError.Text = en ? "Target Error (MSE):" : "Error Objetivo (MSE):";
        BtnResetTrainer.Content = en ? "Reset Weights" : "Reiniciar Pesos";
        BtnTrainerStart.Content = en ? "▶ Start" : "▶ Iniciar";
        BtnTrainerStop.Content = en ? "⏹ Stop" : "⏹ Detener";
        LblLiveInferenceTitle.Text = en ? "LIVE INFERENCE" : "INFERENCIA EN VIVO";
        LblInputValuesHint.Text = en ? "Input values (comma-separated)" : "Valores de entrada (separados por coma)";
        BtnPredict.Content = en ? "Run Prediction" : "Ejecutar Predicción";
        if (LblPredictResult.Text.StartsWith("Result:") || LblPredictResult.Text.StartsWith("Resultado:"))
            LblPredictResult.Text = en ? "Result: -" : "Resultado: -";

        // Comparador
        LblBenchChartTitle.Text = en ? "TIME & CONVERGENCE COMPARISON" : "COMPARATIVA DE TIEMPO Y CONVERGENCIA";
        LblBenchTableTitle.Text = en ? "COMPARATIVE METRICS BY ARCHITECTURE" : "MÉTRICAS COMPARATIVAS POR ARQUITECTURA";
        ColBenchArch.Header = en ? "Architecture" : "Arquitectura";
        ColBenchMse.Header = en ? "Train MSE" : "MSE Train";
        ColBenchTestMse.Header = en ? "Test MSE" : "MSE Test";
        ColBenchEpochs.Header = en ? "Epochs" : "Épocas";
        ColBenchTime.Header = en ? "Total Time" : "Tiempo Total";
        ColBenchMsEpoch.Header = en ? "ms / Epoch" : "ms / Época";
        ColBenchInference.Header = en ? "Infer (µs/sample)" : "Inferencia (µs/muestra)";
        ColBenchOps.Header = en ? "Ops / Spikes" : "Ops / Espigas";
        ColBenchResult.Header = en ? "Result" : "Resultado";
        ChkBenchTrainTestSplit.Content = en ? "80/20 Train/Test Split (Generalization)" : "División 80/20 Train/Test (Generalización)";

        LblBenchConfigTitle.Text = en ? "BENCHMARK SETUP" : "CONFIGURAR COMPARATIVA";
        LblBenchArchsTitle.Text = en ? "ARCHITECTURES TO EVALUATE" : "ARQUITECTURAS A EVALUAR";
        ChkBenchNowral.Content = en ? "Nowral (Static)" : "Nowral (Estática)";
        ChkBenchLibQuid.Content = en ? "LibQuid (Liquid)" : "LibQuid (Líquida)";
        ChkBenchNewNow.Content = en ? "NewNow (Dynamic)" : "NewNow (Dinámica)";
        ChkBenchNeuro.Content = "103KNN Neuromorphic";

        LblBenchDatasetTitle.Text = en ? "COMMON DATASET & TOPOLOGY" : "DATASET Y TOPOLOGÍA COMÚN";
        BtnBenchLoadCsv.Content = en ? "Load CSV" : "Cargar CSV";
        BtnBenchLoadParquet.Content = en ? "Load Parquet" : "Cargar Parquet";
        BtnBenchSynthetic.Content = en ? "Generate Synthetic Dataset (240)" : "Generar Dataset Sintético (240)";
        LblBenchParamsTitle.Text = en ? "TEST PARAMETERS" : "PARÁMETROS DE PRUEBA";
        LblBenchMaxEpochs.Text = en ? "Max Epochs per Model" : "Máximo de Épocas por Modelo";
        LblBenchTargetMse.Text = en ? "Target Error (MSE)" : "Error Objetivo (MSE)";
        BtnRunBenchmark.Content = en ? "▶ Run Benchmark" : "▶ Ejecutar Comparativa";

        TxtActiveModelBadge.Text = FormatArchName(SelectedArchitecture);
        UpdateTopologyStats();
        UpdateTrainerStatusUI();

        DesignerCanvas.SetLanguage(en);
        TrainerCanvas.SetLanguage(en);
        LossChart.SetLanguage(en);
        BenchChart.SetLanguage(en);
    }

    // --- Cambio de Tema Claro / Oscuro ---

    private void BtnThemeToggle_Click(object sender, RoutedEventArgs e)
    {
        _isDarkTheme = !_isDarkTheme;
        ApplyTheme(_isDarkTheme);
    }

    private void ApplyTheme(bool isDark)
    {
        if (isDark)
        {
            Resources["BgWindow"] = new SolidColorBrush(Color.FromRgb(11, 15, 25));
            Resources["BgCard"] = new SolidColorBrush(Color.FromRgb(19, 25, 43));
            Resources["BgElevated"] = new SolidColorBrush(Color.FromRgb(14, 19, 34));
            Resources["BgControl"] = new SolidColorBrush(Color.FromRgb(30, 41, 59));
            Resources["BgInput"] = new SolidColorBrush(Color.FromRgb(13, 16, 26));
            Resources["BorderSubtle"] = new SolidColorBrush(Color.FromRgb(35, 45, 75));
            Resources["BorderControl"] = new SolidColorBrush(Color.FromRgb(51, 65, 85));
            Resources["TextPrimary"] = new SolidColorBrush(Color.FromRgb(248, 250, 252));
            Resources["TextSecondary"] = new SolidColorBrush(Color.FromRgb(148, 163, 184));
            Resources["TextMuted"] = new SolidColorBrush(Color.FromRgb(100, 116, 139));
            Resources["AccentPrimary"] = new SolidColorBrush(Color.FromRgb(56, 189, 248));
            Resources["AccentSuccess"] = new SolidColorBrush(Color.FromRgb(52, 211, 153));
            BtnThemeToggle.Content = _isEnglish ? "☀️ Light Theme" : "☀️ Tema Claro";
        }
        else
        {
            Resources["BgWindow"] = new SolidColorBrush(Color.FromRgb(241, 245, 249));
            Resources["BgCard"] = new SolidColorBrush(Color.FromRgb(255, 255, 255));
            Resources["BgElevated"] = new SolidColorBrush(Color.FromRgb(248, 250, 252));
            Resources["BgControl"] = new SolidColorBrush(Color.FromRgb(226, 232, 240));
            Resources["BgInput"] = new SolidColorBrush(Color.FromRgb(248, 250, 252));
            Resources["BorderSubtle"] = new SolidColorBrush(Color.FromRgb(203, 213, 225));
            Resources["BorderControl"] = new SolidColorBrush(Color.FromRgb(148, 163, 184));
            Resources["TextPrimary"] = new SolidColorBrush(Color.FromRgb(15, 23, 42));
            Resources["TextSecondary"] = new SolidColorBrush(Color.FromRgb(71, 85, 105));
            Resources["TextMuted"] = new SolidColorBrush(Color.FromRgb(100, 116, 139));
            Resources["AccentPrimary"] = new SolidColorBrush(Color.FromRgb(2, 132, 199));
            Resources["AccentSuccess"] = new SolidColorBrush(Color.FromRgb(5, 150, 105));
            BtnThemeToggle.Content = _isEnglish ? "🌙 Dark Theme" : "🌙 Tema Oscuro";
        }

        DesignerCanvas.SetDarkTheme(isDark);
        TrainerCanvas.SetDarkTheme(isDark);
        LossChart.SetDarkTheme(isDark);
        BenchChart.SetDarkTheme(isDark);
    }

    private NetworkArchitectureType SelectedArchitecture => CmbArchitecture.SelectedIndex switch
    {
        0 => NetworkArchitectureType.Nowral,
        1 => NetworkArchitectureType.LibQuid,
        2 => NetworkArchitectureType.NewNow,
        4 => NetworkArchitectureType.FastTree,
        5 => NetworkArchitectureType.FastForest,
        6 => NetworkArchitectureType.StandardMlp,
        7 => NetworkArchitectureType.ResNetTabular,
        8 => NetworkArchitectureType.Lstm,
        _ => NetworkArchitectureType.Neuromorphic
    };

    private string FormatArchName(NetworkArchitectureType arch) => arch switch
    {
        NetworkArchitectureType.Nowral => _isEnglish ? "Nowral (Static)" : "Nowral (Estática)",
        NetworkArchitectureType.LibQuid => _isEnglish ? "LibQuid (Liquid)" : "LibQuid (Líquida)",
        NetworkArchitectureType.NewNow => _isEnglish ? "NewNow (Dynamic)" : "NewNow (Dinámica)",
        NetworkArchitectureType.FastTree => "FastTree (GBDT)",
        NetworkArchitectureType.FastForest => "FastForest (RF)",
        NetworkArchitectureType.StandardMlp => "Standard MLP (PyTorch)",
        NetworkArchitectureType.ResNetTabular => "ResNet-Tabular",
        NetworkArchitectureType.Lstm => "LSTM (Recurrent)",
        _ => "103KNN Neuromorphic"
    };

    private IActivation CreateSelectedActivation() => CmbActivation.SelectedIndex switch
    {
        0 => new ReLUActivation(),
        1 => new SigmoidActivation(),
        _ => new LeakyReLUActivation()
    };

    private void SyncSettingsFromUI()
    {
        _networkSettings.Structure = TxtStructure.Text.Trim();
        _networkSettings.Activation = CmbActivation.SelectedIndex switch
        {
            0 => ActivationType.ReLU,
            1 => ActivationType.Sigmoid,
            _ => ActivationType.LeakyReLU
        };

        if (double.TryParse(TxtLearningRate.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out double lr))
            _trainingSettings.LearningRate = lr;
        if (int.TryParse(TxtBatchSize.Text, out int bs))
            _trainingSettings.BatchSize = Math.Max(1, bs);
        _trainingSettings.Optimizer = CmbOptimizer.SelectedIndex switch
        {
            1 => OptimizerType.SGD,
            2 => OptimizerType.RMSprop,
            _ => OptimizerType.Adam
        };
        if (double.TryParse(TxtStdpRate.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out double stdp))
            _trainingSettings.StdpRate = stdp;
    }

    private void UpdateTopologyStats()
    {
        if (TxtTopologyStats == null || DesignerCanvas == null) return;
        int layerCount = DesignerCanvas.Maps.Count > 0 ? DesignerCanvas.Maps.Count + 1 : 0;
        TxtTopologyStats.Text = _isEnglish
            ? $"Layers: {layerCount}  •  Neurons: {DesignerCanvas.TotalNeurons}  •  Connections: {DesignerCanvas.TotalConnections}"
            : $"Capas: {layerCount}  •  Neuronas: {DesignerCanvas.TotalNeurons}  •  Conexiones: {DesignerCanvas.TotalConnections}";
    }

    // --- Navegación ---

    private void NavDesigner_Checked(object sender, RoutedEventArgs e) => SwitchView(ViewDesigner);
    private void NavTrainer_Checked(object sender, RoutedEventArgs e) => SwitchView(ViewTrainer);
    private void NavBenchmark_Checked(object sender, RoutedEventArgs e)
    {
        SwitchView(ViewBenchmark);
        UpdateDatasetLabels();
    }

    private void SwitchView(Grid target)
    {
        if (ViewDesigner == null || ViewTrainer == null || ViewBenchmark == null) return;
        ViewDesigner.Visibility = target == ViewDesigner ? Visibility.Visible : Visibility.Collapsed;
        ViewTrainer.Visibility = target == ViewTrainer ? Visibility.Visible : Visibility.Collapsed;
        ViewBenchmark.Visibility = target == ViewBenchmark ? Visibility.Visible : Visibility.Collapsed;
    }

    // --- Diseñador ---

    private void CmbArchitecture_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (DesignerCanvas == null || TxtActiveModelBadge == null) return;
        string name = FormatArchName(SelectedArchitecture);
        TxtActiveModelBadge.Text = name;
        DesignerLog(_isEnglish ? $"Selected architecture: {name}" : $"Arquitectura seleccionada: {name}");
    }

    private void BtnShowNetProps_Click(object sender, RoutedEventArgs e)
    {
        PnlNetProperties.Visibility = Visibility.Visible;
        PnlTrainProperties.Visibility = Visibility.Collapsed;
        BtnShowNetProps.SetResourceReference(BorderBrushProperty, "AccentPrimary");
        BtnShowNetProps.SetResourceReference(ForegroundProperty, "AccentPrimary");
        BtnShowTrainProps.SetResourceReference(BorderBrushProperty, "BorderControl");
        BtnShowTrainProps.SetResourceReference(ForegroundProperty, "TextPrimary");
    }

    private void BtnShowTrainProps_Click(object sender, RoutedEventArgs e)
    {
        PnlNetProperties.Visibility = Visibility.Collapsed;
        PnlTrainProperties.Visibility = Visibility.Visible;
        BtnShowTrainProps.SetResourceReference(BorderBrushProperty, "AccentPrimary");
        BtnShowTrainProps.SetResourceReference(ForegroundProperty, "AccentPrimary");
        BtnShowNetProps.SetResourceReference(BorderBrushProperty, "BorderControl");
        BtnShowNetProps.SetResourceReference(ForegroundProperty, "TextPrimary");
    }

    private void TxtStructure_TextChanged(object sender, TextChangedEventArgs e)
    {
        if (DesignerCanvas == null) return;
        if (_structureDebounce == null)
        {
            _structureDebounce = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
            _structureDebounce.Tick += (_, _) =>
            {
                _structureDebounce.Stop();
                SyncSettingsFromUI();
                var layers = _networkSettings.ParseStructure();
                if (layers.Count >= 2)
                {
                    DesignerCanvas.GenerateLayers(layers[0], layers.Skip(1).Take(layers.Count - 2).ToArray(), layers[^1]);
                    UpdateDatasetLabels();
                    DesignerLog(_isEnglish ? $"Structure updated: {string.Join("-", layers)}" : $"Estructura actualizada: {string.Join("-", layers)}");
                }
            };
        }
        _structureDebounce.Stop();
        _structureDebounce.Start();
    }

    private void BtnFullConnect_Click(object sender, RoutedEventArgs e)
    {
        DesignerCanvas.MakeFullConnectivity();
        DesignerLog(_isEnglish ? "All connections restored." : "Todas las conexiones restauradas.");
    }

    public Task BuildNetworkWithAnimation()
    {
        DesignerLog(_isEnglish ? $"Architecture [{FormatArchName(SelectedArchitecture)}] ready." : $"Arquitectura [{FormatArchName(SelectedArchitecture)}] lista.");
        return Task.CompletedTask;
    }

    private void BtnStartTraining_Click(object sender, RoutedEventArgs e) => OpenTrainerView();

    public void OpenTrainerView()
    {
        if (DesignerCanvas.Maps.Count == 0) return;

        SyncSettingsFromUI();
        var arch = SelectedArchitecture;
        var activation = CreateSelectedActivation();

        LossChart.Clear();
        _trainerEngine.Initialize(arch, DesignerCanvas.Maps, activation, _trainingSettings);
        TrainerCanvas.LoadExistingStructure(_trainerEngine.Maps);

        string name = FormatArchName(arch);
        LblTrainerTitle.Text = name.ToUpperInvariant();
        LblTrainerOpsTitle.Text = arch == NetworkArchitectureType.Neuromorphic
            ? (_isEnglish ? "SPIKES / SAMPLE" : "ESPIGAS / MUESTRA")
            : (_isEnglish ? "OPS / INFERENCE" : "OPS / INFERENCIA");
        TxtActiveModelBadge.Text = name;
        NavTrainer.IsChecked = true;
        UpdateTrainerStatusUI();
    }

    // --- Entrenador ---

    private async void BtnTrainerStart_Click(object sender, RoutedEventArgs e)
    {
        LossChart.Clear();
        float targetError = float.TryParse(TxtTargetError.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out float te)
            ? te
            : 0.0001f;
        await _trainerEngine.StartTrainingAsync(targetError);
    }

    private void BtnTrainerStop_Click(object sender, RoutedEventArgs e) => _trainerEngine.Stop();

    private void BtnResetTrainer_Click(object sender, RoutedEventArgs e)
    {
        SyncSettingsFromUI();
        LossChart.Clear();
        _trainerEngine.Initialize(SelectedArchitecture, DesignerCanvas.Maps, CreateSelectedActivation(), _trainingSettings);
        TrainerCanvas.LoadExistingStructure(_trainerEngine.Maps);
        TrainerLog(_isEnglish ? "Weights reset." : "Pesos reiniciados.");
    }

    private void BtnLoadCsv_Click(object sender, RoutedEventArgs e)
    {
        var ofd = new OpenFileDialog { Filter = "CSV Files (*.csv)|*.csv|All files (*.*)|*.*" };
        if (ofd.ShowDialog() != true) return;
        try
        {
            EnsureMapsInitialized();
            _trainerEngine.LoadCsv(ofd.FileName);
            UpdateTrainerStatusUI();
            UpdateDatasetLabels();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"CSV error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void BtnLoadParquet_Click(object sender, RoutedEventArgs e)
    {
        var ofd = new OpenFileDialog { Filter = "Parquet Files (*.parquet)|*.parquet" };
        if (ofd.ShowDialog() != true) return;
        try
        {
            EnsureMapsInitialized();
            _trainerEngine.LoadParquet(ofd.FileName);
            UpdateTrainerStatusUI();
            UpdateDatasetLabels();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Parquet error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void EnsureMapsInitialized()
    {
        if (DesignerCanvas.Maps.Count == 0) return;

        bool mismatch = _trainerEngine.Maps.Count != DesignerCanvas.Maps.Count;
        if (!mismatch)
        {
            for (int i = 0; i < DesignerCanvas.Maps.Count; i++)
            {
                if (_trainerEngine.Maps[i].SourceSize != DesignerCanvas.Maps[i].SourceSize ||
                    _trainerEngine.Maps[i].TargetSize != DesignerCanvas.Maps[i].TargetSize ||
                    _trainerEngine.Maps[i].Connections.Count != DesignerCanvas.Maps[i].Connections.Count)
                {
                    mismatch = true;
                    break;
                }
            }
        }

        if (mismatch)
        {
            SyncSettingsFromUI();
            _trainerEngine.Initialize(SelectedArchitecture, DesignerCanvas.Maps, CreateSelectedActivation(), _trainingSettings);
        }
    }

    private void BtnSaveModel_Click(object sender, RoutedEventArgs e)
    {
        var sfd = new SaveFileDialog { Filter = "Model JSON (*.json)|*.json", DefaultExt = "json" };
        if (sfd.ShowDialog() != true) return;
        try
        {
            string basePath = sfd.FileName.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
                ? sfd.FileName[..^5]
                : sfd.FileName;
            _trainerEngine.SaveModel(basePath);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Save error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void BtnLoadModel_Click(object sender, RoutedEventArgs e)
    {
        var ofd = new OpenFileDialog { Filter = "Model JSON (*.json)|*.json" };
        if (ofd.ShowDialog() != true) return;
        try
        {
            _trainerEngine.LoadModel(ofd.FileName);
            LossChart.Clear();
            UpdateTrainerStatusUI();
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Load error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void BtnPredict_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var raw = TxtPredictInput.Text.Split(new[] { ',', ';', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(s => float.Parse(s.Trim(), CultureInfo.InvariantCulture))
                .ToArray();

            var res = _trainerEngine.Predict(raw);
            string prefix = _isEnglish ? "Result" : "Resultado";
            LblPredictResult.Text = res.Success && res.Prediction != null
                ? $"{prefix}: [{string.Join(", ", res.Prediction.Select(v => v.ToString("F4")))}]"
                : $"Error: {res.Error}";
        }
        catch
        {
            LblPredictResult.Text = _isEnglish
                ? "Error: Invalid format (use comma-separated numbers)."
                : "Error: Formato inválido (usa valores separados por coma).";
        }
    }

    // --- Comparador de Arquitecturas ---

    private void BtnBenchSynthetic_Click(object sender, RoutedEventArgs e)
    {
        if (DesignerCanvas.Maps.Count == 0) return;
        int inDim = DesignerCanvas.Maps[0].SourceSize;
        int outDim = DesignerCanvas.Maps[^1].TargetSize;
        var (inputs, targets) = ModelTrainerEngine.GenerateSyntheticDataset(inDim, outDim, sampleCount: 240);
        _trainerEngine.SetData(inputs, targets);
        UpdateTrainerStatusUI();
        UpdateDatasetLabels();
    }

    private async void BtnRunBenchmark_Click(object sender, RoutedEventArgs e)
    {
        await ExecuteBenchmarkCoreAsync();
    }

    private async Task ExecuteBenchmarkCoreAsync()
    {
        if (DesignerCanvas.Maps.Count == 0)
        {
            MessageBox.Show(_isEnglish ? "Define the network structure first." : "Define la estructura de red primero.", "Info", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        if (_trainerEngine.RawInputs == null || _trainerEngine.RawTargets == null)
        {
            BtnBenchSynthetic_Click(this, new RoutedEventArgs());
        }

        var selectedArchs = new List<NetworkArchitectureType>();
        if (ChkBenchNowral.IsChecked == true) selectedArchs.Add(NetworkArchitectureType.Nowral);
        if (ChkBenchLibQuid.IsChecked == true) selectedArchs.Add(NetworkArchitectureType.LibQuid);
        if (ChkBenchNewNow.IsChecked == true) selectedArchs.Add(NetworkArchitectureType.NewNow);
        if (ChkBenchNeuro.IsChecked == true) selectedArchs.Add(NetworkArchitectureType.Neuromorphic);
        if (ChkBenchFastTree.IsChecked == true) selectedArchs.Add(NetworkArchitectureType.FastTree);
        if (ChkBenchFastForest.IsChecked == true) selectedArchs.Add(NetworkArchitectureType.FastForest);
        if (ChkBenchStdMlp.IsChecked == true) selectedArchs.Add(NetworkArchitectureType.StandardMlp);
        if (ChkBenchResNet.IsChecked == true) selectedArchs.Add(NetworkArchitectureType.ResNetTabular);
        if (ChkBenchLstm.IsChecked == true) selectedArchs.Add(NetworkArchitectureType.Lstm);

        if (selectedArchs.Count == 0)
        {
            MessageBox.Show(_isEnglish ? "Select at least one architecture." : "Selecciona al menos una arquitectura.", "Info", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        SyncSettingsFromUI();
        long maxEpochs = long.TryParse(TxtBenchEpochs.Text, out long ep) ? Math.Max(1, ep) : 40;
        float targetMse = float.TryParse(TxtBenchTargetError.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out float tm) ? tm : 0.0001f;

        var rawInputs = _trainerEngine.RawInputs!;
        var rawTargets = _trainerEngine.RawTargets!;
        var maps = DesignerCanvas.Maps;
        var activation = CreateSelectedActivation();

        bool useSplit = ChkBenchTrainTestSplit.IsChecked == true && rawInputs.Length >= 10;
        float[][] trainInputs = rawInputs;
        float[][] trainTargets = rawTargets;
        float[][] evalInputs = rawInputs;
        float[][] evalTargets = rawTargets;

        if (useSplit)
        {
            var splitRng = new Random(103);
            int[] perm = Enumerable.Range(0, rawInputs.Length).OrderBy(_ => splitRng.Next()).ToArray();
            int trainCount = Math.Max(5, (int)(rawInputs.Length * 0.8));
            trainInputs = perm.Take(trainCount).Select(idx => rawInputs[idx]).ToArray();
            trainTargets = perm.Take(trainCount).Select(idx => rawTargets[idx]).ToArray();
            evalInputs = perm.Skip(trainCount).Select(idx => rawInputs[idx]).ToArray();
            evalTargets = perm.Skip(trainCount).Select(idx => rawTargets[idx]).ToArray();
        }

        _benchmarkResults.Clear();
        BenchChart.SetResults(_benchmarkResults);
        _benchCts = new CancellationTokenSource();

        BtnRunBenchmark.IsEnabled = false;
        BtnStopBenchmark.IsEnabled = true;

        try
        {
            foreach (var arch in selectedArchs)
            {
                if (_benchCts.Token.IsCancellationRequested) break;

                var item = new BenchmarkResultItem
                {
                    Architecture = FormatArchName(arch),
                    Status = _isEnglish ? "Training..." : "Entrenando..."
                };
                _benchmarkResults.Add(item);

                using var benchEngine = new ModelTrainerEngine();
                benchEngine.Initialize(arch, maps, activation, _trainingSettings);
                benchEngine.SetData(trainInputs, trainTargets, normalize: true);

                using var cancelReg = _benchCts.Token.Register(() => benchEngine.Stop());
                benchEngine.OnProgressUpdated += () =>
                {
                    if (!benchEngine.IsTraining) return;
                    long ep = benchEngine.CurrentEpoch;
                    if (ep != 1 && ep % 20 != 0 && ep < maxEpochs) return;
                    Dispatcher.BeginInvoke(() =>
                    {
                        if (!benchEngine.IsTraining) return;
                        item.EpochsRun = benchEngine.CurrentEpoch;
                        item.FinalMse = benchEngine.CurrentMSE;
                        item.Status = _isEnglish
                            ? $"Training ({benchEngine.CurrentEpoch}/{maxEpochs})..."
                            : $"Entrenando ({benchEngine.CurrentEpoch}/{maxEpochs})...";
                        BenchDataGrid.Items.Refresh();
                    });
                };

                var sw = Stopwatch.StartNew();
                await benchEngine.StartTrainingAsync(targetMse, maxEpochs);
                sw.Stop();

                if (_benchCts.Token.IsCancellationRequested)
                {
                    item.EpochsRun = benchEngine.CurrentEpoch;
                    item.FinalMse = benchEngine.CurrentMSE;
                    item.TotalMs = sw.Elapsed.TotalMilliseconds;
                    item.Status = _isEnglish ? "Stopped" : "Detenido";
                    BenchDataGrid.Items.Refresh();
                    break;
                }

                float testMse = float.NaN;
                var infSw = Stopwatch.StartNew();
                int evalCount = 0;
                await Task.Run(() =>
                {
                    testMse = benchEngine.EvaluateNormalizedMse(evalInputs, evalTargets);
                    for (int i = 0; i < evalInputs.Length; i++)
                    {
                        var pred = benchEngine.Predict(evalInputs[i]);
                        if (pred.Success) evalCount++;
                    }
                });
                infSw.Stop();

                item.FinalMse = benchEngine.CurrentMSE;
                item.TestMse = float.IsNaN(testMse) ? null : testMse;
                item.EpochsRun = benchEngine.CurrentEpoch;
                item.TotalMs = sw.Elapsed.TotalMilliseconds;
                item.MsPerEpoch = item.EpochsRun > 0 ? item.TotalMs / item.EpochsRun : item.TotalMs;
                item.InferenceUsPerSample = evalCount > 0 ? infSw.Elapsed.TotalMicroseconds / evalCount : 0;
                item.OpsOrSpikes = benchEngine.LastOpsOrSpikes;
                item.Status = item.FinalMse <= targetMse
                    ? (_isEnglish ? "Target reached" : "Objetivo alcanzado")
                    : (_isEnglish ? "Completed" : "Completado");

                BenchDataGrid.Items.Refresh();
                BenchChart.SetResults(_benchmarkResults);
            }

            if (_benchmarkResults.Count > 1)
            {
                var bestMse = _benchmarkResults.OrderBy(r => r.TestMse ?? r.FinalMse).First();
                var fastest = _benchmarkResults.OrderBy(r => r.TotalMs).First();
                foreach (var r in _benchmarkResults)
                {
                    var tags = new List<string>();
                    if (r == bestMse) tags.Add(_isEnglish ? "Lowest MSE" : "Menor MSE");
                    if (r == fastest) tags.Add(_isEnglish ? "Fastest" : "Más rápido");
                    if (tags.Count > 0) r.Status = string.Join(" + ", tags);
                }
                BenchDataGrid.Items.Refresh();
            }
        }
        finally
        {
            BtnRunBenchmark.IsEnabled = true;
            BtnStopBenchmark.IsEnabled = false;
        }
    }

    private void BtnStopBenchmark_Click(object sender, RoutedEventArgs e) => _benchCts?.Cancel();

    private void UpdateDatasetLabels()
    {
        if (LblBenchDataset != null)
        {
            LblBenchDataset.Text = _isEnglish
                ? $"Active dataset: {_trainerEngine.SampleCount} samples"
                : $"Dataset activo: {_trainerEngine.SampleCount} muestras";
        }
        if (LblBenchTopology != null && DesignerCanvas.Maps.Count > 0)
        {
            var layers = new List<int> { DesignerCanvas.Maps[0].SourceSize };
            layers.AddRange(DesignerCanvas.Maps.Select(m => m.TargetSize));
            LblBenchTopology.Text = _isEnglish
                ? $"Topology: {string.Join("-", layers)} ({DesignerCanvas.TotalConnections} connections) | LR: {_trainingSettings.LearningRate}"
                : $"Topología: {string.Join("-", layers)} ({DesignerCanvas.TotalConnections} conexiones) | LR: {_trainingSettings.LearningRate}";
        }
    }

    private void UpdateTrainerStatusUI()
    {
        LblTrainerSamples.Text = _isEnglish
            ? $"{_trainerEngine.SampleCount} samples"
            : $"{_trainerEngine.SampleCount} muestras";
        LblTrainerEpoch.Text = $"{_trainerEngine.CurrentEpoch:N0}";
        LblTrainerOps.Text = $"{_trainerEngine.LastOpsOrSpikes:N0}";
        LblTrainerMSE.Text = $"{_trainerEngine.CurrentMSE:F6}";
        UpdateDatasetLabels();
    }

    private void DesignerLog(string msg)
    {
        TxtDesignerLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {msg}{Environment.NewLine}");
        TxtDesignerLog.ScrollToEnd();
    }

    private void TrainerLog(string msg)
    {
        TxtTrainerLog.AppendText($"[{DateTime.Now:HH:mm:ss}] {msg}{Environment.NewLine}");
        TxtTrainerLog.ScrollToEnd();
    }

    // --- API HTTP Local ---

    public object GetRemoteStatus() => new
    {
        status = "ok",
        networkType = SelectedArchitecture.ToString(),
        hasTrainer = _trainerEngine.Maps.Count > 0,
        layers = DesignerCanvas.Maps.Count + 1,
        connections = DesignerCanvas.TotalConnections,
        samples = _trainerEngine.SampleCount
    };

    public void SwitchViewByName(string view)
    {
        switch (view.ToLowerInvariant())
        {
            case "designer":
                NavDesigner.IsChecked = true;
                break;
            case "trainer":
                NavTrainer.IsChecked = true;
                break;
            case "benchmark":
                NavBenchmark.IsChecked = true;
                break;
        }
    }

    public object LoadRemoteCsv(string path)
    {
        EnsureMapsInitialized();
        _trainerEngine.LoadCsv(path);
        UpdateTrainerStatusUI();
        UpdateDatasetLabels();
        return new { status = "ok", samples = _trainerEngine.SampleCount, path };
    }

    public object GenerateRemoteSynthetic(int sampleCount)
    {
        if (DesignerCanvas.Maps.Count == 0)
            return new { status = "error", message = "No topology defined" };
        int inDim = DesignerCanvas.Maps[0].SourceSize;
        int outDim = DesignerCanvas.Maps[^1].TargetSize;
        var (inputs, targets) = ModelTrainerEngine.GenerateSyntheticDataset(inDim, outDim, Math.Max(10, sampleCount));
        _trainerEngine.SetData(inputs, targets);
        UpdateTrainerStatusUI();
        UpdateDatasetLabels();
        return new { status = "ok", samples = _trainerEngine.SampleCount };
    }

    public async Task<object> RunRemoteBenchmarkAsync(BenchmarkRunRequest req)
    {
        var tcs = new TaskCompletionSource<object>();
        await Dispatcher.InvokeAsync(async () =>
        {
            try
            {
                NavBenchmark.IsChecked = true;
                if (req.LearningRate.HasValue)
                    TxtLearningRate.Text = req.LearningRate.Value.ToString(CultureInfo.InvariantCulture);
                if (req.BatchSize.HasValue)
                    TxtBatchSize.Text = req.BatchSize.Value.ToString();
                TxtBenchEpochs.Text = req.MaxEpochs.ToString();
                TxtBenchTargetError.Text = req.TargetError.ToString(CultureInfo.InvariantCulture);

                var set = new HashSet<int>(req.Architectures ?? new[] { 0, 1, 2, 3, 4, 5 });
                ChkBenchNowral.IsChecked = set.Contains(0);
                ChkBenchLibQuid.IsChecked = set.Contains(1);
                ChkBenchNewNow.IsChecked = set.Contains(2);
                ChkBenchNeuro.IsChecked = set.Contains(3);
                ChkBenchFastTree.IsChecked = set.Contains(4);
                ChkBenchFastForest.IsChecked = set.Contains(5);
                ChkBenchStdMlp.IsChecked = set.Contains(6);
                ChkBenchResNet.IsChecked = set.Contains(7);
                ChkBenchLstm.IsChecked = set.Contains(8);
                ChkBenchTrainTestSplit.IsChecked = req.TrainTestSplit;

                await ExecuteBenchmarkCoreAsync();
                tcs.SetResult(GetRemoteBenchmarkResults());
            }
            catch (Exception ex)
            {
                tcs.SetResult(new { status = "error", message = ex.Message });
            }
        });
        return await tcs.Task;
    }

    public object GetRemoteBenchmarkResults() => new
    {
        status = "ok",
        topology = LblBenchTopology.Text,
        samples = _trainerEngine.SampleCount,
        results = _benchmarkResults.ToArray()
    };

    public void ApplyRemoteDesign(int type, int inputs, int[] hidden, int outputs)
    {
        CmbArchitecture.SelectedIndex = Math.Clamp(type, 0, 8);
        var list = new List<int> { inputs };
        list.AddRange(hidden);
        list.Add(outputs);
        TxtStructure.Text = string.Join(" ", list);
        DesignerCanvas.GenerateLayers(inputs, hidden, outputs);
        DesignerCanvas.MakeFullConnectivity();
        _trainerEngine.Initialize(SelectedArchitecture, DesignerCanvas.Maps, CreateSelectedActivation(), _trainingSettings);
        UpdateDatasetLabels();
        DesignerLog($"Remote design applied: {string.Join("-", list)} ({FormatArchName(SelectedArchitecture)}).");
    }

    public void SetRemoteData(float[][] inputs, float[][] targets)
    {
        _trainerEngine.SetData(inputs, targets);
        UpdateTrainerStatusUI();
    }

    public void ExecuteRemoteTraining()
    {
        LossChart.Clear();
        float targetError = float.TryParse(TxtTargetError.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out float te)
            ? te
            : 0.0001f;
        _ = _trainerEngine.StartTrainingAsync(targetError);
    }

    public object GetTrainingProgress() => new
    {
        status = "ok",
        epoch = _trainerEngine.CurrentEpoch,
        mse = _trainerEngine.CurrentMSE,
        isTraining = _trainerEngine.IsTraining
    };

    public void StopRemoteTraining() => _trainerEngine.Stop();

    public object ExecuteRemotePredict(float[] inputData) => _trainerEngine.Predict(inputData);
}

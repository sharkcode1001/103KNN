using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace _103KNN.Studio;

public sealed class BenchmarkResultItem
{
    public string Architecture { get; set; } = "";
    public float FinalMse { get; set; }
    public long EpochsRun { get; set; }
    public double TotalMs { get; set; }
    public double MsPerEpoch { get; set; }
    public double InferenceUsPerSample { get; set; }
    public long OpsOrSpikes { get; set; }
    public string Status { get; set; } = "";

    public string FinalMseDisplay => $"{FinalMse:F6}";
    public string TotalMsDisplay => $"{TotalMs:F1} ms";
    public string MsPerEpochDisplay => $"{MsPerEpoch:F2} ms";
    public string InferenceDisplay => $"{InferenceUsPerSample:F1} µs";
    public string OpsDisplay => $"{OpsOrSpikes:N0}";
}

/// <summary>
/// Gráfico comparativo de barras horizontales para visualizar tiempo (ms) y MSE por arquitectura.
/// </summary>
public class BenchmarkBarChart : Canvas
{
    private List<BenchmarkResultItem> _items = new();
    private bool _isDarkTheme = false;
    private bool _isEnglish = true;

    private Brush ChartBg => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(13, 16, 26))
        : new SolidColorBrush(Color.FromRgb(248, 250, 252));
    private Brush TextBrush => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(226, 232, 240))
        : new SolidColorBrush(Color.FromRgb(30, 41, 59));
    private Brush SubTextBrush => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(148, 163, 184))
        : new SolidColorBrush(Color.FromRgb(100, 116, 139));
    private Brush TrackBrush => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(30, 41, 59))
        : new SolidColorBrush(Color.FromRgb(226, 232, 240));

    private static readonly Brush[] BarColors =
    {
        new SolidColorBrush(Color.FromRgb(56, 189, 248)),  // Sky
        new SolidColorBrush(Color.FromRgb(129, 140, 248)), // Indigo
        new SolidColorBrush(Color.FromRgb(246, 177, 122)), // Amber
        new SolidColorBrush(Color.FromRgb(52, 211, 153))   // Emerald
    };

    public BenchmarkBarChart()
    {
        Background = ChartBg;
        ClipToBounds = true;
        SizeChanged += (_, _) => Redraw();
    }

    public void SetDarkTheme(bool isDark)
    {
        _isDarkTheme = isDark;
        Background = ChartBg;
        Redraw();
    }

    public void SetLanguage(bool isEnglish)
    {
        _isEnglish = isEnglish;
        Redraw();
    }

    public void SetResults(IEnumerable<BenchmarkResultItem> items)
    {
        _items = items.ToList();
        Redraw();
    }

    public void Redraw()
    {
        Children.Clear();
        double w = ActualWidth > 60 ? ActualWidth : 640;
        double h = ActualHeight > 40 ? ActualHeight : 190;

        if (_items.Count == 0)
        {
            var empty = new TextBlock
            {
                Text = _isEnglish
                    ? "Run a benchmark to visualize the relative performance of each architecture"
                    : "Ejecuta una comparativa para visualizar el rendimiento relativo de cada arquitectura",
                Foreground = SubTextBrush,
                FontSize = 12,
                FontStyle = FontStyles.Italic
            };
            SetLeft(empty, Math.Max(16, (w - 430) * 0.5));
            SetTop(empty, h * 0.45);
            Children.Add(empty);
            return;
        }

        double maxMs = Math.Max(1.0, _items.Max(i => i.TotalMs));
        const double leftCol = 155;
        const double rightPad = 170;
        double barMaxW = Math.Max(60, w - leftCol - rightPad);

        double rowH = Math.Min(42, (h - 24) / _items.Count);
        double startY = 14;

        for (int i = 0; i < _items.Count; i++)
        {
            var item = _items[i];
            double y = startY + i * rowH;

            var nameTb = new TextBlock
            {
                Text = item.Architecture,
                Foreground = TextBrush,
                FontSize = 12,
                FontWeight = FontWeights.SemiBold
            };
            SetLeft(nameTb, 12);
            SetTop(nameTb, y + 6);
            Children.Add(nameTb);

            var track = new Border
            {
                Width = barMaxW,
                Height = 18,
                CornerRadius = new CornerRadius(6),
                Background = TrackBrush
            };
            SetLeft(track, leftCol);
            SetTop(track, y + 5);
            Children.Add(track);

            double ratio = Math.Clamp(item.TotalMs / maxMs, 0.03, 1.0);
            var bar = new Border
            {
                Width = barMaxW * ratio,
                Height = 18,
                CornerRadius = new CornerRadius(6),
                Background = BarColors[i % BarColors.Length]
            };
            SetLeft(bar, leftCol);
            SetTop(bar, y + 5);
            Children.Add(bar);

            var valTb = new TextBlock
            {
                Text = $"{item.TotalMs:F0} ms | MSE {item.FinalMse:F4}",
                Foreground = SubTextBrush,
                FontSize = 11,
                FontFamily = new FontFamily("Consolas")
            };
            SetLeft(valTb, leftCol + barMaxW + 10);
            SetTop(valTb, y + 7);
            Children.Add(valTb);
        }
    }
}

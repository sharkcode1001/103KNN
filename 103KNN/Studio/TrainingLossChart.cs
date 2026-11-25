using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace _103KNN.Studio;

/// <summary>
/// Gráfico moderno en tiempo real de la curva de error (MSE) durante el entrenamiento.
/// </summary>
public class TrainingLossChart : Canvas
{
    private readonly List<(long Epoch, float Mse)> _points = new();
    private const int MaxStoredPoints = 300;
    private bool _isDarkTheme = false;
    private bool _isEnglish = true;

    private Brush ChartBg => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(13, 16, 26))
        : new SolidColorBrush(Color.FromRgb(248, 250, 252));
    private Brush GridStroke => _isDarkTheme
        ? new SolidColorBrush(Color.FromArgb(45, 148, 163, 184))
        : new SolidColorBrush(Color.FromArgb(70, 148, 163, 184));
    private static readonly Brush LineStroke = new SolidColorBrush(Color.FromRgb(2, 132, 199));
    private Brush AxisTextBrush => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(148, 163, 184))
        : new SolidColorBrush(Color.FromRgb(71, 85, 105));
    private Brush EmptyTextBrush => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(100, 116, 139))
        : new SolidColorBrush(Color.FromRgb(100, 116, 139));
    private static readonly Brush AreaFill = new LinearGradientBrush(
        Color.FromArgb(95, 56, 189, 248),
        Color.FromArgb(5, 56, 189, 248),
        90);

    public TrainingLossChart()
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

    public void AddPoint(long epoch, float mse)
    {
        if (float.IsNaN(mse) || float.IsInfinity(mse)) return;

        _points.Add((epoch, mse));
        if (_points.Count > MaxStoredPoints)
        {
            // Submuestreo ligero para mantener historial completo sin saturar
            _points.RemoveAt(0);
        }
        Redraw();
    }

    public void Clear()
    {
        _points.Clear();
        Redraw();
    }

    public void Redraw()
    {
        Children.Clear();

        double w = ActualWidth > 40 ? ActualWidth : 600;
        double h = ActualHeight > 40 ? ActualHeight : 180;

        const double padLeft = 58;
        const double padRight = 18;
        const double padTop = 20;
        const double padBottom = 24;

        double plotW = Math.Max(40, w - padLeft - padRight);
        double plotH = Math.Max(40, h - padTop - padBottom);

        // Cuadrícula horizontal
        for (int i = 0; i <= 4; i++)
        {
            double y = padTop + (plotH * i / 4.0);
            var gridLine = new Line
            {
                X1 = padLeft,
                Y1 = y,
                X2 = w - padRight,
                Y2 = y,
                Stroke = GridStroke,
                StrokeThickness = 1,
                StrokeDashArray = new DoubleCollection { 3, 3 },
                IsHitTestVisible = false
            };
            Children.Add(gridLine);
        }

        if (_points.Count < 2)
        {
            var empty = new TextBlock
            {
                Text = _isEnglish
                    ? "Start training to visualize the convergence curve (MSE)"
                    : "Inicia el entrenamiento para visualizar la curva de convergencia (MSE)",
                Foreground = EmptyTextBrush,
                FontSize = 12,
                FontStyle = FontStyles.Italic
            };
            SetLeft(empty, padLeft + Math.Max(10, (plotW - 380) * 0.5));
            SetTop(empty, padTop + plotH * 0.45);
            Children.Add(empty);
            return;
        }

        float minMse = _points.Min(p => p.Mse);
        float maxMse = _points.Max(p => p.Mse);
        float range = Math.Max(1e-7f, maxMse - minMse);

        // Etiquetas eje Y
        AddAxisLabel($"{maxMse:F5}", 4, padTop - 7);
        AddAxisLabel($"{(minMse + range * 0.5f):F5}", 4, padTop + plotH * 0.5 - 7);
        AddAxisLabel($"{minMse:F5}", 4, padTop + plotH - 7);

        // Etiquetas eje X
        string epWord = _isEnglish ? "Epoch" : "Época";
        AddAxisLabel($"{epWord} {_points[0].Epoch}", padLeft, h - 18);
        AddAxisLabel($"{epWord} {_points[^1].Epoch}", w - padRight - 65, h - 18);

        var linePoints = new PointCollection(_points.Count);
        var areaFigure = new PathFigure();

        for (int i = 0; i < _points.Count; i++)
        {
            double x = padLeft + (plotW * i / (_points.Count - 1.0));
            double normY = (_points[i].Mse - minMse) / range;
            double y = padTop + plotH * (1.0 - normY);
            var pt = new Point(x, y);
            linePoints.Add(pt);

            if (i == 0)
                areaFigure.StartPoint = new Point(x, padTop + plotH);
            areaFigure.Segments.Add(new LineSegment(pt, true));
        }

        areaFigure.Segments.Add(new LineSegment(new Point(padLeft + plotW, padTop + plotH), true));
        areaFigure.IsClosed = true;

        var areaGeo = new PathGeometry();
        areaGeo.Figures.Add(areaFigure);

        Children.Add(new Path
        {
            Data = areaGeo,
            Fill = AreaFill,
            IsHitTestVisible = false
        });

        Children.Add(new Polyline
        {
            Points = linePoints,
            Stroke = LineStroke,
            StrokeThickness = 2.2,
            StrokeLineJoin = PenLineJoin.Round,
            IsHitTestVisible = false
        });

        // Punto final destacado
        var lastPt = linePoints[^1];
        var dot = new Ellipse
        {
            Width = 7,
            Height = 7,
            Fill = Brushes.White,
            Stroke = LineStroke,
            StrokeThickness = 2,
            IsHitTestVisible = false
        };
        SetLeft(dot, lastPt.X - 3.5);
        SetTop(dot, lastPt.Y - 3.5);
        Children.Add(dot);
    }

    private void AddAxisLabel(string text, double x, double y)
    {
        var tb = new TextBlock
        {
            Text = text,
            Foreground = AxisTextBrush,
            FontSize = 10,
            FontFamily = new FontFamily("Consolas"),
            IsHitTestVisible = false
        };
        SetLeft(tb, x);
        SetTop(tb, y);
        Children.Add(tb);
    }
}

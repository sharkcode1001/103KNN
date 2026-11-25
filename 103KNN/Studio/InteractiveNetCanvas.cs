using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Effects;
using System.Windows.Shapes;
using NowLib;

namespace _103KNN.Studio;

/// <summary>
/// Lienzo interactivo moderno para diseñar y visualizar topologías de redes neuronales.
/// Dibuja conexiones suaves mediante curvas de Bézier y nodos codificados por color según su capa.
/// </summary>
public class InteractiveNetCanvas : Canvas
{
    private readonly List<Point[]> _nodePositions = new();
    public List<ConnectionMap> Maps { get; private set; } = new();
    public event EventHandler? StructureChanged;

    private (int layer, int index)? _selectedNode;
    private (int layer, int index)? _hoveredNode;

    private bool _isDarkTheme = false;
    private bool _isEnglish = true;

    private Brush CanvasBg => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(13, 16, 26))
        : new SolidColorBrush(Color.FromRgb(248, 250, 252));
    private Brush GridDotBrush => _isDarkTheme
        ? new SolidColorBrush(Color.FromArgb(40, 148, 163, 184))
        : new SolidColorBrush(Color.FromArgb(65, 100, 116, 139));

    private static readonly Brush InputColor = new SolidColorBrush(Color.FromRgb(56, 189, 248));   // Sky Cyan
    private static readonly Brush HiddenColor = new SolidColorBrush(Color.FromRgb(129, 140, 248)); // Indigo
    private static readonly Brush OutputColor = new SolidColorBrush(Color.FromRgb(52, 211, 153));  // Emerald
    private static readonly Brush SelectedColor = new SolidColorBrush(Color.FromRgb(246, 177, 122)); // Amber
    private Brush NodeInnerFill => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(18, 22, 36))
        : new SolidColorBrush(Color.FromRgb(255, 255, 255));

    private Brush ConnectionNormal => _isDarkTheme
        ? new SolidColorBrush(Color.FromArgb(75, 129, 140, 248))
        : new SolidColorBrush(Color.FromArgb(95, 99, 102, 241));
    private static readonly Brush ConnectionHighlight = new SolidColorBrush(Color.FromArgb(220, 56, 189, 248));
    private Brush LabelTextBrush => _isDarkTheme
        ? new SolidColorBrush(Color.FromRgb(148, 163, 184))
        : new SolidColorBrush(Color.FromRgb(51, 65, 85));
    private Brush PillBgBrush => _isDarkTheme
        ? new SolidColorBrush(Color.FromArgb(180, 30, 36, 56))
        : new SolidColorBrush(Color.FromArgb(220, 255, 255, 255));

    public int TotalNeurons => _nodePositions.Sum(l => l.Length);
    public int TotalConnections => Maps.Sum(m => m.Connections.Count);

    public InteractiveNetCanvas()
    {
        Background = CanvasBg;
        ClipToBounds = true;
        SizeChanged += (_, _) => RecalculateAndRedraw();
    }

    public void SetDarkTheme(bool isDark)
    {
        _isDarkTheme = isDark;
        Background = CanvasBg;
        Redraw();
    }

    public void SetLanguage(bool isEnglish)
    {
        _isEnglish = isEnglish;
        Redraw();
    }

    public void GenerateLayers(int inputs, int[] hidden, int outputs)
    {
        var layers = new List<int> { inputs };
        if (hidden != null) layers.AddRange(hidden);
        layers.Add(outputs);
        UpdateStructure(layers);
    }

    public void UpdateStructure(List<int> layers)
    {
        if (layers.Count < 2) return;

        Maps.Clear();
        for (int i = 0; i < layers.Count - 1; i++)
        {
            var map = new ConnectionMap(layers[i], layers[i + 1]);
            map.FillFullConnectivity();
            Maps.Add(map);
        }

        _selectedNode = null;
        _hoveredNode = null;
        CalculatePositions(layers);
        StructureChanged?.Invoke(this, EventArgs.Empty);
        Redraw();
    }

    public void MakeFullConnectivity()
    {
        foreach (var m in Maps) m.FillFullConnectivity();
        StructureChanged?.Invoke(this, EventArgs.Empty);
        Redraw();
    }

    public void LoadExistingStructure(List<ConnectionMap> maps)
    {
        if (maps == null || maps.Count == 0) return;
        Maps = maps;
        _selectedNode = null;
        _hoveredNode = null;
        RecalculateAndRedraw();
    }

    private void RecalculateAndRedraw()
    {
        if (Maps.Count == 0) return;
        var layers = new List<int> { Maps[0].SourceSize };
        foreach (var m in Maps) layers.Add(m.TargetSize);
        CalculatePositions(layers);
        Redraw();
    }

    private void CalculatePositions(List<int> layers)
    {
        _nodePositions.Clear();
        if (layers.Count == 0) return;

        double w = ActualWidth > 40 ? ActualWidth : 740;
        double h = ActualHeight > 40 ? ActualHeight : 380;

        double topMargin = 38;
        double bottomMargin = 20;
        double usableH = Math.Max(60, h - topMargin - bottomMargin);

        double xStep = w / (layers.Count + 1.0);
        for (int l = 0; l < layers.Count; l++)
        {
            int count = layers[l];
            var layerNodes = new Point[count];
            double yStep = usableH / (count + 1.0);
            for (int i = 0; i < count; i++)
            {
                layerNodes[i] = new Point(xStep * (l + 1), topMargin + yStep * (i + 1));
            }
            _nodePositions.Add(layerNodes);
        }
    }

    public void Redraw()
    {
        Children.Clear();
        DrawSubtleGrid();
        if (_nodePositions.Count == 0) return;

        var activeNode = _selectedNode ?? _hoveredNode;

        // 1. Conexiones (curvas Bézier suaves)
        for (int mIdx = 0; mIdx < Maps.Count; mIdx++)
        {
            if (mIdx + 1 >= _nodePositions.Count) continue;
            var map = Maps[mIdx];

            foreach (var c in map.Connections)
            {
                if (c.Source >= _nodePositions[mIdx].Length || c.Target >= _nodePositions[mIdx + 1].Length)
                    continue;

                var p1 = _nodePositions[mIdx][c.Source];
                var p2 = _nodePositions[mIdx + 1][c.Target];

                bool isHighlighted = activeNode.HasValue &&
                    ((activeNode.Value.layer == mIdx && activeNode.Value.index == c.Source) ||
                     (activeNode.Value.layer == mIdx + 1 && activeNode.Value.index == c.Target));

                double midX = (p1.X + p2.X) * 0.5;
                var geometry = new PathGeometry();
                var figure = new PathFigure { StartPoint = p1 };
                figure.Segments.Add(new BezierSegment(
                    new Point(midX, p1.Y),
                    new Point(midX, p2.Y),
                    p2,
                    true));
                geometry.Figures.Add(figure);

                var path = new Path
                {
                    Data = geometry,
                    Stroke = isHighlighted ? ConnectionHighlight : ConnectionNormal,
                    StrokeThickness = isHighlighted ? 1.8 : 0.95,
                    IsHitTestVisible = false
                };
                Children.Add(path);
            }
        }

        // 2. Capas y neuronas
        for (int l = 0; l < _nodePositions.Count; l++)
        {
            int count = _nodePositions[l].Length;
            double radius = count > 24 ? 4.5 : count > 12 ? 6.5 : 8.5;
            double diameter = radius * 2;

            Brush accent = l == 0 ? InputColor
                : l == _nodePositions.Count - 1 ? OutputColor
                : HiddenColor;

            string layerName = l == 0 ? (_isEnglish ? $"Input ({count})" : $"Entrada ({count})")
                : l == _nodePositions.Count - 1 ? (_isEnglish ? $"Output ({count})" : $"Salida ({count})")
                : (_isEnglish ? $"Hidden {l} ({count})" : $"Oculta {l} ({count})");

            var badge = new Border
            {
                Background = PillBgBrush,
                BorderBrush = accent,
                BorderThickness = new Thickness(1),
                CornerRadius = new CornerRadius(10),
                Padding = new Thickness(8, 2, 8, 2),
                IsHitTestVisible = false,
                Child = new TextBlock
                {
                    Text = layerName,
                    Foreground = LabelTextBrush,
                    FontSize = 10.5,
                    FontWeight = FontWeights.SemiBold
                }
            };
            SetLeft(badge, _nodePositions[l][0].X - 36);
            SetTop(badge, 8);
            Children.Add(badge);

            for (int i = 0; i < count; i++)
            {
                var p = _nodePositions[l][i];
                bool isSelected = _selectedNode.HasValue && _selectedNode.Value.layer == l && _selectedNode.Value.index == i;
                bool isHovered = _hoveredNode.HasValue && _hoveredNode.Value.layer == l && _hoveredNode.Value.index == i;

                if (isSelected || isHovered)
                {
                    var halo = new Ellipse
                    {
                        Width = diameter + 10,
                        Height = diameter + 10,
                        Fill = isSelected
                            ? new SolidColorBrush(Color.FromArgb(65, 246, 177, 122))
                            : new SolidColorBrush(Color.FromArgb(55, 56, 189, 248)),
                        IsHitTestVisible = false
                    };
                    SetLeft(halo, p.X - radius - 5);
                    SetTop(halo, p.Y - radius - 5);
                    Children.Add(halo);
                }

                var ellipse = new Ellipse
                {
                    Width = diameter,
                    Height = diameter,
                    Fill = isSelected ? SelectedColor : NodeInnerFill,
                    Stroke = isSelected ? Brushes.White : accent,
                    StrokeThickness = isSelected ? 2.2 : 1.8,
                    IsHitTestVisible = false
                };

                SetLeft(ellipse, p.X - radius);
                SetTop(ellipse, p.Y - radius);
                Children.Add(ellipse);
            }
        }
    }

    private void DrawSubtleGrid()
    {
        double w = ActualWidth > 0 ? ActualWidth : 740;
        double h = ActualHeight > 0 ? ActualHeight : 380;
        const double step = 28;

        for (double x = step; x < w; x += step)
        {
            for (double y = step; y < h; y += step)
            {
                var dot = new Ellipse
                {
                    Width = 1.6,
                    Height = 1.6,
                    Fill = GridDotBrush,
                    IsHitTestVisible = false
                };
                SetLeft(dot, x);
                SetTop(dot, y);
                Children.Add(dot);
            }
        }
    }

    protected override void OnMouseMove(MouseEventArgs e)
    {
        base.OnMouseMove(e);
        var hit = HitTest(e.GetPosition(this));
        if (hit != _hoveredNode)
        {
            _hoveredNode = hit;
            Cursor = hit.HasValue ? Cursors.Hand : Cursors.Arrow;
            Redraw();
        }
    }

    protected override void OnMouseLeave(MouseEventArgs e)
    {
        base.OnMouseLeave(e);
        if (_hoveredNode != null)
        {
            _hoveredNode = null;
            Cursor = Cursors.Arrow;
            Redraw();
        }
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonDown(e);
        var clickedNode = HitTest(e.GetPosition(this));

        if (clickedNode != null)
        {
            if (_selectedNode == null)
            {
                _selectedNode = clickedNode;
            }
            else
            {
                int layer1 = _selectedNode.Value.layer;
                int node1 = _selectedNode.Value.index;
                int layer2 = clickedNode.Value.layer;
                int node2 = clickedNode.Value.index;

                if (layer2 == layer1 + 1 && layer1 < Maps.Count)
                {
                    Maps[layer1].ToggleConnection(node1, node2);
                    StructureChanged?.Invoke(this, EventArgs.Empty);
                    _selectedNode = null;
                }
                else
                {
                    _selectedNode = clickedNode;
                }
            }
        }
        else
        {
            _selectedNode = null;
        }

        Redraw();
    }

    private (int layer, int index)? HitTest(Point mouseLoc)
    {
        const double radiusSq = 12 * 12;
        for (int l = 0; l < _nodePositions.Count; l++)
        {
            for (int i = 0; i < _nodePositions[l].Length; i++)
            {
                var p = _nodePositions[l][i];
                double dx = mouseLoc.X - p.X;
                double dy = mouseLoc.Y - p.Y;
                if (dx * dx + dy * dy <= radiusSq) return (l, i);
            }
        }
        return null;
    }
}

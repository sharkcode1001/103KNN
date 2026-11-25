using TorchSharp;
using static TorchSharp.torch;

namespace NowLib;

public class ConnectionRecord
{
    public int Source { get; set; }
    public int Target { get; set; }
    public ConnectionRecord() { }
    public ConnectionRecord(int src, int dst) { Source = src; Target = dst; }
    public override bool Equals(object? obj) => obj is ConnectionRecord o && Source == o.Source && Target == o.Target;
    public override int GetHashCode() => HashCode.Combine(Source, Target);
}

public class ConnectionMap
{
    public int SourceSize { get; set; }
    public int TargetSize { get; set; }
    public List<ConnectionRecord> Connections { get; set; } = new();

    public ConnectionMap() { }
    public ConnectionMap(int sourceSize, int targetSize) { SourceSize = sourceSize; TargetSize = targetSize; }

    public void AddConnection(int src, int dst)
    {
        var c = new ConnectionRecord(src, dst);
        if (!Connections.Contains(c)) Connections.Add(c);
    }

    public void FillFullConnectivity()
    {
        Connections.Clear();
        for (int i = 0; i < SourceSize; i++)
            for (int j = 0; j < TargetSize; j++)
                AddConnection(i, j);
    }

    public Tensor GenerateMask()
    {
        var mask = zeros(new long[] { TargetSize, SourceSize }, dtype: ScalarType.Float32);
        foreach (var c in Connections) mask[c.Target, c.Source] = 1.0f;
        return mask;
    }

    public void ToggleConnection(int src, int dst)
    {
        var c = new ConnectionRecord(src, dst);
        if (Connections.Contains(c)) Connections.Remove(c);
        else Connections.Add(c);
    }
}
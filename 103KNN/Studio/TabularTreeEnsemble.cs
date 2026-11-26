using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace _103KNN.Studio;

public sealed class TabularInputRow
{
    public float[] Features { get; set; } = Array.Empty<float>();
    public float Label { get; set; }
}

public sealed class TabularScoreOutput
{
    [ColumnName("Score")]
    public float Score { get; set; }
}

/// <summary>
/// Envoltorio multi-salida para modelos tabulares estándar de ML.NET:
/// FastTree (Gradient Boosted Decision Trees) y FastForest (Random Forest).
/// </summary>
public sealed class TabularTreeEnsemble : IDisposable
{
    private readonly MLContext _ml = new(seed: 103);
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly bool _isRandomForest;
    private readonly List<ITransformer> _models = new();
    private readonly List<PredictionEngine<TabularInputRow, TabularScoreOutput>> _engines = new();

    public int NumberOfTrees { get; set; }

    public TabularTreeEnsemble(int inputDim, int outputDim, bool isRandomForest, int numberOfTrees = 200)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _isRandomForest = isRandomForest;
        NumberOfTrees = numberOfTrees;
    }

    public float Train(float[][] inputs, float[][] targets, double learningRate = 0.1)
    {
        DisposeEngines();
        _models.Clear();

        var schema = SchemaDefinition.Create(typeof(TabularInputRow));
        schema[nameof(TabularInputRow.Features)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, _inputDim);

        for (int k = 0; k < _outputDim; k++)
        {
            var rows = new List<TabularInputRow>(inputs.Length);
            for (int i = 0; i < inputs.Length; i++)
            {
                rows.Add(new TabularInputRow
                {
                    Features = inputs[i],
                    Label = targets[i][k]
                });
            }

            var dataView = _ml.Data.LoadFromEnumerable(rows, schema);

            IEstimator<ITransformer> trainer = _isRandomForest
                ? _ml.Regression.Trainers.FastForest(new FastForestRegressionTrainer.Options
                {
                    NumberOfTrees = NumberOfTrees,
                    NumberOfLeaves = 20,
                    MinimumExampleCountPerLeaf = 1
                })
                : _ml.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options
                {
                    NumberOfTrees = NumberOfTrees,
                    NumberOfLeaves = 20,
                    MinimumExampleCountPerLeaf = 1,
                    LearningRate = Math.Clamp(learningRate * 20.0, 0.05, 0.4)
                });

            var model = trainer.Fit(dataView);
            _models.Add(model);
            _engines.Add(_ml.Model.CreatePredictionEngine<TabularInputRow, TabularScoreOutput>(model, inputSchemaDefinition: schema));
        }

        // Calcular MSE real sobre el conjunto entrenado
        double totalSqErr = 0;
        var rowReuse = new TabularInputRow();
        for (int i = 0; i < inputs.Length; i++)
        {
            rowReuse.Features = inputs[i];
            for (int k = 0; k < _outputDim; k++)
            {
                float pred = _engines[k].Predict(rowReuse).Score;
                double err = pred - targets[i][k];
                totalSqErr += err * err;
            }
        }

        return (float)(totalSqErr / (inputs.Length * _outputDim));
    }

    public float[] Predict(float[] input)
    {
        var output = new float[_outputDim];
        var row = new TabularInputRow { Features = input };
        for (int k = 0; k < _outputDim; k++)
        {
            output[k] = _engines[k].Predict(row).Score;
        }
        return output;
    }

    private void DisposeEngines()
    {
        foreach (var eng in _engines) eng.Dispose();
        _engines.Clear();
    }

    public void Dispose() => DisposeEngines();
}

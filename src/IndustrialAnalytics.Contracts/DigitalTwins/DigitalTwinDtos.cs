using System.Text.Json.Serialization;

namespace IndustrialAnalytics.Contracts.DigitalTwins;

/// <summary>
/// MODEL OUTPUT — unsupervised anomaly detector (Experiment 3,
/// Isolation Forest). <see cref="Score"/> uses sklearn's
/// <c>score_samples</c> convention: HIGHER is more normal, LOWER is more
/// anomalous, and a window is anomalous when the score is BELOW
/// <see cref="Threshold"/>.
/// </summary>
public sealed record TwinAnomalyDto(
    bool IsAnomalous,
    double Score,
    double Threshold,
    string ScoreDirection =
        "score_samples: HIGHER = more normal, LOWER = more anomalous; ANOMALOUS when score < threshold"
);

/// <summary>
/// MODEL OUTPUT — supervised fault classifier (Experiment 2, Random
/// Forest). <see cref="Probabilities"/> is the full class distribution
/// when the estimator supports it.
/// </summary>
public sealed record TwinClassificationDto(
    string PredictedClass,
    double? Confidence,
    IReadOnlyDictionary<string, double>? Probabilities
);

/// <summary>Engineered features — the frozen 7-feature streaming contract.</summary>
public sealed record TwinFeaturesDto(
    double VibrationRms,
    double VibrationStd,
    double VibrationPeak,
    double VibrationPeakToPeak,
    double VibrationKurtosis,
    double VibrationSkewness,
    double CrestFactor
);

/// <summary>Drive-reported operating point. An input to the classifier only.</summary>
public sealed record TwinOperatingContextDto(
    double? MotorLoadHp,
    double? RotationalSpeedRpm
);

/// <summary>Provenance of the 2048-sample window the state was derived from.</summary>
public sealed record TwinWindowDto(
    long WindowStartSequence,
    long WindowEndSequence,
    int SampleCount
);

/// <summary>
/// DEMO GROUND TRUTH ONLY — the simulated scenario behind the replay.
/// <para>
/// This is NOT a model output and is NOT an inference input. It exists so
/// a demonstration can be scored against reality. A real asset cannot
/// report its own fault class or defect diameter — that is precisely what
/// the models are asked to infer. It is null whenever demo mode is off.
/// </para>
/// </summary>
public sealed record TwinGroundTruthDto(
    string? RecordingId,
    string? FaultClass,
    double? FaultSeverityIn
);

/// <summary>
/// Current Digital Twin state for one asset.
/// <para>
/// The grouping is deliberate: <see cref="Anomaly"/> and
/// <see cref="Classification"/> are MODEL OUTPUT,
/// <see cref="OperatingContext"/> and <see cref="Features"/> are inputs,
/// and <see cref="DemoGroundTruth"/> is never either.
/// </para>
/// Raw 2048-sample vibration arrays are deliberately NOT exposed here —
/// only the window's sequence range and its engineered features.
/// </summary>
public sealed record DigitalTwinStateDto(
    string AssetId,
    DateTime LastUpdatedUtc,
    TwinWindowDto Window,
    TwinAnomalyDto Anomaly,
    TwinClassificationDto Classification,
    TwinFeaturesDto Features,
    TwinOperatingContextDto OperatingContext,
    [property: JsonPropertyName("demoGroundTruth")] TwinGroundTruthDto? DemoGroundTruth
);

public sealed record DigitalTwinListResponse(
    int Count,
    IReadOnlyList<DigitalTwinStateDto> Items
);

/// <summary>One historical inference for an asset, newest first.</summary>
public sealed record DigitalTwinHistoryPointDto(
    long InferenceId,
    DateTime InferredAtUtc,
    long WindowStartSequence,
    long WindowEndSequence,
    bool IsAnomalous,
    double AnomalyScore,
    string PredictedClass,
    double? Confidence,
    double VibrationRms,
    double VibrationKurtosis,
    [property: JsonPropertyName("demoFaultClass")] string? DemoFaultClass
);

public sealed record DigitalTwinHistoryResponse(
    string AssetId,
    int Count,
    IReadOnlyList<DigitalTwinHistoryPointDto> Items
);

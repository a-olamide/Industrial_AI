using System.Text.Json.Serialization;

namespace IndustrialAnalytics.Contracts.DigitalTwins;

/// <summary>
/// A compact summary of recent inference history, included so the
/// explanation can say whether the current window is representative or an
/// outlier. Counts only — no per-window detail, no ground truth.
/// </summary>
public sealed record TwinTrendDto(
    int WindowsConsidered,
    int WindowsFlaggedAnomalous,
    IReadOnlyDictionary<string, int> PredictedClassCounts
);

/// <summary>
/// The EXACT payload sent to Claude. Nothing outside this record reaches
/// the model.
/// <para>
/// Deliberately absent, and asserted by tests: raw 2048-sample vibration
/// arrays, the CWRU filename, <c>recordingId</c>, <c>demoFaultClass</c>,
/// <c>demoFaultSeverityIn</c> — any demo ground truth. Ground truth is a
/// classroom evaluation aid; letting it reach the model would make the
/// explanation a restatement of the answer key rather than a reading of
/// the evidence, and would invalidate the MOTOR_003 demonstration.
/// </para>
/// </summary>
public sealed record ClaudeExplanationRequestDto(
    string AssetId,
    DateTime ObservedAtUtc,
    TwinAnomalyDto Anomaly,
    TwinClassificationDto Classification,
    TwinFeaturesDto Features,
    TwinOperatingContextDto OperatingContext,
    TwinTrendDto? RecentTrend
);

/// <summary>
/// The structured explanation Claude returns, constrained by a JSON schema.
/// <para>
/// This is an AI-generated <i>explanation of</i> model output — not an
/// independent diagnosis, and never a substitute for either model's
/// prediction.
/// </para>
/// </summary>
public sealed record MaintenanceExplanationDto(
    string Summary,
    string LikelyCondition,
    IReadOnlyList<string> Evidence,
    IReadOnlyList<string> RecommendedActions,
    string ConfidenceNote
);

/// <summary>
/// API envelope. Echoes the authoritative ML verdict alongside the prose so
/// a caller can always see what the models actually said, independently of
/// how Claude phrased it.
/// </summary>
public sealed record MaintenanceExplanationResponseDto(
    string AssetId,
    long WindowEndSequence,
    DateTime GeneratedAtUtc,
    string Model,
    bool FromCache,
    [property: JsonPropertyName("mlVerdict")] MlVerdictDto MlVerdict,
    MaintenanceExplanationDto Explanation
);

/// <summary>The authoritative model output, restated for the client.</summary>
public sealed record MlVerdictDto(
    bool IsAnomalous,
    double AnomalyScore,
    string PredictedClass,
    double? Confidence
);

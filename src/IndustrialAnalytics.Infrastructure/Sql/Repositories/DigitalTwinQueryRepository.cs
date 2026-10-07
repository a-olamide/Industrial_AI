using System.Text.Json;
using Dapper;
using IndustrialAnalytics.Contracts.DigitalTwins;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    /// <summary>
    /// Reads dbo.asset_twin_current and dbo.asset_twin_inference_history.
    ///
    /// Those tables are populated by ml/streaming (Spark -> JDBC). This
    /// repository never writes to them, mirroring how
    /// <see cref="RiskQueryRepository"/> reads the Spark-owned risk tables.
    ///
    /// Ground truth lives in demo_* columns and is projected into a
    /// separate DTO that the caller can suppress entirely, so it can
    /// never be confused with a model output.
    /// </summary>
    public sealed class DigitalTwinQueryRepository(ISqlConnectionFactory f) : IDigitalTwinQueryRepository
    {
        private const string CurrentColumns = @"
                asset_id                 AS AssetId,
                last_updated_utc         AS LastUpdatedUtc,
                window_start_sequence    AS WindowStartSequence,
                window_end_sequence      AS WindowEndSequence,
                sample_count             AS SampleCount,
                is_anomalous             AS IsAnomalous,
                anomaly_score            AS AnomalyScore,
                anomaly_threshold        AS AnomalyThreshold,
                predicted_class          AS PredictedClass,
                confidence               AS Confidence,
                class_probabilities_json AS ClassProbabilitiesJson,
                vibration_rms            AS VibrationRms,
                vibration_std            AS VibrationStd,
                vibration_peak           AS VibrationPeak,
                vibration_peak_to_peak   AS VibrationPeakToPeak,
                vibration_kurtosis       AS VibrationKurtosis,
                vibration_skewness       AS VibrationSkewness,
                crest_factor             AS CrestFactor,
                motor_load_hp            AS MotorLoadHp,
                rotational_speed_rpm     AS RotationalSpeedRpm,
                demo_recording_id        AS DemoRecordingId,
                demo_fault_class         AS DemoFaultClass,
                demo_fault_severity_in   AS DemoFaultSeverityIn";

        private sealed class CurrentRow
        {
            public string AssetId { get; init; } = "";
            public DateTime LastUpdatedUtc { get; init; }
            public long WindowStartSequence { get; init; }
            public long WindowEndSequence { get; init; }
            public int SampleCount { get; init; }
            public bool IsAnomalous { get; init; }
            public double AnomalyScore { get; init; }
            public double AnomalyThreshold { get; init; }
            public string PredictedClass { get; init; } = "";
            public double? Confidence { get; init; }
            public string? ClassProbabilitiesJson { get; init; }
            public double VibrationRms { get; init; }
            public double VibrationStd { get; init; }
            public double VibrationPeak { get; init; }
            public double VibrationPeakToPeak { get; init; }
            public double VibrationKurtosis { get; init; }
            public double VibrationSkewness { get; init; }
            public double CrestFactor { get; init; }
            public double? MotorLoadHp { get; init; }
            public double? RotationalSpeedRpm { get; init; }
            public string? DemoRecordingId { get; init; }
            public string? DemoFaultClass { get; init; }
            public double? DemoFaultSeverityIn { get; init; }
        }

        private sealed class HistoryRow
        {
            public long InferenceId { get; init; }
            public DateTime InferredAtUtc { get; init; }
            public long WindowStartSequence { get; init; }
            public long WindowEndSequence { get; init; }
            public bool IsAnomalous { get; init; }
            public double AnomalyScore { get; init; }
            public string PredictedClass { get; init; } = "";
            public double? Confidence { get; init; }
            public double VibrationRms { get; init; }
            public double VibrationKurtosis { get; init; }
            public string? DemoFaultClass { get; init; }
        }

        private static IReadOnlyDictionary<string, double>? ParseProbabilities(string? json)
        {
            if (string.IsNullOrWhiteSpace(json)) return null;
            try
            {
                return JsonSerializer.Deserialize<Dictionary<string, double>>(json);
            }
            catch (JsonException)
            {
                // A malformed probability blob must not take down the twin
                // read path; the prediction itself is still usable.
                return null;
            }
        }

        private static DigitalTwinStateDto Map(CurrentRow r, bool includeGroundTruth)
        {
            TwinGroundTruthDto? truth = null;
            if (includeGroundTruth &&
                (r.DemoRecordingId is not null || r.DemoFaultClass is not null))
            {
                truth = new TwinGroundTruthDto(
                    r.DemoRecordingId, r.DemoFaultClass, r.DemoFaultSeverityIn);
            }

            return new DigitalTwinStateDto(
                r.AssetId,
                DateTime.SpecifyKind(r.LastUpdatedUtc, DateTimeKind.Utc),
                new TwinWindowDto(r.WindowStartSequence, r.WindowEndSequence, r.SampleCount),
                new TwinAnomalyDto(r.IsAnomalous, r.AnomalyScore, r.AnomalyThreshold),
                new TwinClassificationDto(
                    r.PredictedClass, r.Confidence, ParseProbabilities(r.ClassProbabilitiesJson)),
                new TwinFeaturesDto(
                    r.VibrationRms, r.VibrationStd, r.VibrationPeak, r.VibrationPeakToPeak,
                    r.VibrationKurtosis, r.VibrationSkewness, r.CrestFactor),
                new TwinOperatingContextDto(r.MotorLoadHp, r.RotationalSpeedRpm),
                truth);
        }

        public async Task<IReadOnlyList<DigitalTwinStateDto>> GetAllAsync(
            bool includeGroundTruth, CancellationToken ct)
        {
            var sql = $"SELECT {CurrentColumns} FROM dbo.asset_twin_current ORDER BY asset_id;";

            using var conn = f.Create();
            var rows = await conn.QueryAsync<CurrentRow>(
                new CommandDefinition(sql, cancellationToken: ct));
            return rows.Select(r => Map(r, includeGroundTruth)).ToList();
        }

        public async Task<DigitalTwinStateDto?> GetByAssetAsync(
            string assetId, bool includeGroundTruth, CancellationToken ct)
        {
            var sql = $@"SELECT {CurrentColumns}
                         FROM dbo.asset_twin_current
                         WHERE asset_id = @assetId;";

            using var conn = f.Create();
            var row = await conn.QuerySingleOrDefaultAsync<CurrentRow>(
                new CommandDefinition(sql, new { assetId }, cancellationToken: ct));
            return row is null ? null : Map(row, includeGroundTruth);
        }

        public async Task<IReadOnlyList<DigitalTwinHistoryPointDto>> GetHistoryAsync(
            string assetId, int take, bool includeGroundTruth, CancellationToken ct)
        {
            const string sql = @"
                SELECT TOP (@take)
                    inference_id           AS InferenceId,
                    inferred_at_utc        AS InferredAtUtc,
                    window_start_sequence  AS WindowStartSequence,
                    window_end_sequence    AS WindowEndSequence,
                    is_anomalous           AS IsAnomalous,
                    anomaly_score          AS AnomalyScore,
                    predicted_class        AS PredictedClass,
                    confidence             AS Confidence,
                    vibration_rms          AS VibrationRms,
                    vibration_kurtosis     AS VibrationKurtosis,
                    demo_fault_class       AS DemoFaultClass
                FROM dbo.asset_twin_inference_history
                WHERE asset_id = @assetId
                ORDER BY window_end_sequence DESC, inference_id DESC;";

            using var conn = f.Create();
            var rows = await conn.QueryAsync<HistoryRow>(
                new CommandDefinition(sql, new { assetId, take }, cancellationToken: ct));

            return rows.Select(r => new DigitalTwinHistoryPointDto(
                r.InferenceId,
                DateTime.SpecifyKind(r.InferredAtUtc, DateTimeKind.Utc),
                r.WindowStartSequence,
                r.WindowEndSequence,
                r.IsAnomalous,
                r.AnomalyScore,
                r.PredictedClass,
                r.Confidence,
                r.VibrationRms,
                r.VibrationKurtosis,
                includeGroundTruth ? r.DemoFaultClass : null
            )).ToList();
        }
    }
}

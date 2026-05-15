using System.Diagnostics;
using IndustrialTelemetry.Producer.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace IndustrialTelemetry.Producer.Services;

public sealed class TelemetrySimulator
{
    private const int QualityGood = 192;      // OPC UA: Good (Non-specific)
    private const int QualityUncertain = 64;  // OPC UA: Uncertain (Non-specific)

    private readonly Random _rng = new();
    private readonly Stopwatch _uptime = Stopwatch.StartNew();
    private readonly ILogger<TelemetrySimulator> _logger;
    private readonly AnomalyOptions _anomalyOpts;

    // Per-asset active anomaly: type + UTC expiry
    private readonly Dictionary<string, (AnomalyType Type, DateTimeOffset ExpiresAt)> _activeAnomalies = new();

    public IReadOnlyList<AssetProfile> Assets { get; } = BuildProfiles();

    public TelemetrySimulator(ILogger<TelemetrySimulator> logger, IOptions<AnomalyOptions> anomalyOptions)
    {
        _logger = logger;
        _anomalyOpts = anomalyOptions.Value;
    }

    public IEnumerable<TelemetryEvent> GenerateEvents(AssetProfile asset)
    {
        var ts = DateTimeOffset.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ");
        var minutesElapsed = _uptime.Elapsed.TotalMinutes;
        var anomalyType = UpdateAnomalyState(asset);

        var tagValues = ComputeTagValues(asset, minutesElapsed, anomalyType);
        var faultCode = ComputeFaultCode(tagValues);

        yield return Make(asset, ts, "run_state", 1.0, QualityGood);
        yield return Make(asset, ts, "fault_code", faultCode, faultCode > 0 ? QualityUncertain : QualityGood);

        // Always emit quality=192 for sensor tags so Spark never filters anomaly readings.
        foreach (var (tag, value) in tagValues)
            yield return Make(asset, ts, tag, Math.Round(value, 3), QualityGood);
    }

    // ── Anomaly state machine ────────────────────────────────────────────────

    private AnomalyType UpdateAnomalyState(AssetProfile asset)
    {
        if (_activeAnomalies.TryGetValue(asset.AssetId, out var current))
        {
            if (DateTimeOffset.UtcNow < current.ExpiresAt)
                return current.Type;

            _logger.LogInformation(
                "ANOMALY END   | asset={AssetId} | type={Type}",
                asset.AssetId, current.Type);
            _activeAnomalies.Remove(asset.AssetId);
        }

        // Roll the dice — only start a new anomaly when none is active
        if (_rng.NextDouble() >= _anomalyOpts.Probability)
            return AnomalyType.None;

        var type = PickAnomalyType();
        var expiry = DateTimeOffset.UtcNow.AddSeconds(_anomalyOpts.DurationSeconds);
        _activeAnomalies[asset.AssetId] = (type, expiry);

        _logger.LogWarning(
            "ANOMALY START | asset={AssetId} | type={Type} | duration={Dur}s",
            asset.AssetId, type, _anomalyOpts.DurationSeconds);

        return type;
    }

    private AnomalyType PickAnomalyType()
        // AnomalyType values 1–6 map to the six anomaly types; 0 = None
        => (AnomalyType)(_rng.Next(6) + 1);

    // ── Tag value computation ────────────────────────────────────────────────

    private Dictionary<string, double> ComputeTagValues(AssetProfile asset, double minutesElapsed, AnomalyType anomaly)
    {
        var values = new Dictionary<string, double>(asset.TagSpecs.Count);

        foreach (var spec in asset.TagSpecs)
            values[spec.Name] = spec.DegradedValue(minutesElapsed) + Gaussian(0, spec.NoiseAmplitude);

        if (asset.HealthState == AssetHealthState.VibrationSpike)
            ApplyVibrationSpike(values, minutesElapsed);

        if (anomaly != AnomalyType.None)
            ApplyAnomaly(values, anomaly);

        return values;
    }

    private void ApplyAnomaly(Dictionary<string, double> values, AnomalyType anomaly)
    {
        switch (anomaly)
        {
            case AnomalyType.Overheating:
                // 92–100 °C  — Spark critical threshold: > 88
                values["temp_bearing_c"] = 92.0 + _rng.NextDouble() * 8.0;
                break;

            case AnomalyType.HighVibration:
                // 8–12 mm/s  — Spark critical threshold: > 7
                values["vib_rms_mm_s"] = 8.0 + _rng.NextDouble() * 4.0;
                break;

            case AnomalyType.HighCurrent:
                // 32–38 A    — Spark critical threshold: > 30
                values["motor_current_a"] = 32.0 + _rng.NextDouble() * 6.0;
                break;

            case AnomalyType.LowFlow:
                // 55–65 m³/h — Spark critical threshold: < 70
                values["flow_rate_m3_h"] = 55.0 + _rng.NextDouble() * 10.0;
                break;

            case AnomalyType.LowDischargePressure:
                // 4–5 bar    — Spark critical threshold: < 5.5
                values["discharge_pressure_bar"] = 4.0 + _rng.NextDouble() * 1.0;
                break;

            case AnomalyType.CombinedFailure:
                values["temp_bearing_c"]        = 92.0 + _rng.NextDouble() * 8.0;
                values["vib_rms_mm_s"]           = 8.0  + _rng.NextDouble() * 4.0;
                values["motor_current_a"]        = 32.0 + _rng.NextDouble() * 6.0;
                values["flow_rate_m3_h"]         = 55.0 + _rng.NextDouble() * 10.0;
                values["discharge_pressure_bar"] = 4.0  + _rng.NextDouble() * 1.0;
                break;
        }
    }

    private void ApplyVibrationSpike(Dictionary<string, double> values, double minutesElapsed)
    {
        // Spike probability grows from 5 % to 25 % as degradation progresses over 30 min.
        var degradationFactor = Math.Min(1.0, minutesElapsed / 30.0);
        if (_rng.NextDouble() >= 0.05 + 0.20 * degradationFactor) return;

        var magnitude = 5.0 + _rng.NextDouble() * 8.0;  // 5–13 mm/s
        values["vib_rms_mm_s"] = magnitude;
        values["motor_current_a"] += magnitude * 0.35;   // sympathetic current rise
    }

    // ── Fault code: OPC-style bitmask ────────────────────────────────────────
    // Bit 0 (1)  – High bearing temperature  > 80 °C
    // Bit 1 (2)  – High vibration            > 5 mm/s
    // Bit 2 (4)  – High motor current        > 28 A
    // Bit 3 (8)  – Low flow rate             < 80 m³/h
    // Bit 4 (16) – Low discharge pressure    < 6 bar

    private static int ComputeFaultCode(Dictionary<string, double> v)
    {
        var code = 0;
        if (v.TryGetValue("temp_bearing_c",         out var t)   && t   > 80.0) code |= 1;
        if (v.TryGetValue("vib_rms_mm_s",           out var vib) && vib > 5.0)  code |= 2;
        if (v.TryGetValue("motor_current_a",        out var i)   && i   > 28.0) code |= 4;
        if (v.TryGetValue("flow_rate_m3_h",         out var f)   && f   < 80.0) code |= 8;
        if (v.TryGetValue("discharge_pressure_bar", out var p)   && p   < 6.0)  code |= 16;
        return code;
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    private double Gaussian(double mean, double stdDev)
    {
        // Box-Muller transform — produces normally distributed noise
        var u1 = 1.0 - _rng.NextDouble();
        var u2 = 1.0 - _rng.NextDouble();
        return mean + stdDev * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private static TelemetryEvent Make(AssetProfile a, string ts, string tag, double value, int quality)
        => new()
        {
            AssetId = a.AssetId,
            AssetType = a.AssetType,
            Ts = ts,
            Tag = tag,
            Value = value,
            Quality = quality
        };

    // ── Asset profiles ───────────────────────────────────────────────────────

    private static IReadOnlyList<AssetProfile> BuildProfiles() =>
    [
        // PUMP_001 — Bearing wear: rising temperature + vibration, falling flow
        new()
        {
            AssetId = "PUMP_001",
            AssetType = "Pump",
            HealthState = AssetHealthState.BearingWear,
            TagSpecs =
            [
                new("temp_bearing_c",         55.0, 0.80, +28.0, 60.0),
                new("vib_rms_mm_s",            2.5, 0.20,  +4.5, 60.0),
                new("motor_current_a",         16.2, 0.40,  +3.0, 60.0),
                new("flow_rate_m3_h",         115.0, 2.50, -22.0, 60.0),
                new("discharge_pressure_bar",   8.2, 0.12,  -1.5, 60.0),
            ]
        },

        // PUMP_002 — Healthy: all values within nominal operating band
        new()
        {
            AssetId = "PUMP_002",
            AssetType = "Pump",
            HealthState = AssetHealthState.Healthy,
            TagSpecs =
            [
                new("temp_bearing_c",         44.0, 0.40),
                new("vib_rms_mm_s",            1.0, 0.08),
                new("motor_current_a",         14.5, 0.25),
                new("flow_rate_m3_h",         125.0, 1.80),
                new("discharge_pressure_bar",   8.8, 0.09),
            ]
        },

        // COMP_001 — Overheating: rapid thermal and current escalation
        new()
        {
            AssetId = "COMP_001",
            AssetType = "Compressor",
            HealthState = AssetHealthState.Overheating,
            TagSpecs =
            [
                new("temp_bearing_c",         82.0, 1.20, +32.0, 45.0),
                new("vib_rms_mm_s",            1.8, 0.15,  +0.8, 45.0),
                new("motor_current_a",         22.0, 0.60,  +9.0, 45.0),
                new("flow_rate_m3_h",          92.0, 2.00, -12.0, 45.0),
                new("discharge_pressure_bar",  10.5, 0.20,  +1.2, 45.0),
            ]
        },

        // FAN_001 — Vibration spikes: stochastic high-amplitude transients
        new()
        {
            AssetId = "FAN_001",
            AssetType = "Fan",
            HealthState = AssetHealthState.VibrationSpike,
            TagSpecs =
            [
                new("temp_bearing_c",         52.0, 0.90, +5.0, 45.0),
                new("vib_rms_mm_s",            1.8, 0.15),
                new("motor_current_a",         18.0, 0.50, +1.5, 45.0),
                new("flow_rate_m3_h",         110.0, 2.50),
                new("discharge_pressure_bar",   7.2, 0.12),
            ]
        },

        // GBX_001 — Current drift: slow monotonic rise in motor load
        new()
        {
            AssetId = "GBX_001",
            AssetType = "Gearbox",
            HealthState = AssetHealthState.CurrentDrift,
            TagSpecs =
            [
                new("temp_bearing_c",         48.0, 0.60, +14.0, 90.0),
                new("vib_rms_mm_s",            1.4, 0.12,  +0.8, 90.0),
                new("motor_current_a",         20.0, 0.50, +16.0, 90.0),
                new("flow_rate_m3_h",         108.0, 2.00,  -8.0, 90.0),
                new("discharge_pressure_bar",   7.8, 0.10,  -0.5, 90.0),
            ]
        },
    ];
}

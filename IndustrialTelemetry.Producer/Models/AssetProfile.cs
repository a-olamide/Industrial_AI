namespace IndustrialTelemetry.Producer.Models;

public enum AssetHealthState
{
    Healthy,
    BearingWear,
    Overheating,
    VibrationSpike,
    CurrentDrift
}

/// <summary>
/// Defines the simulation behaviour for a single tag on an asset.
/// DegradationDelta is the total value shift applied once fully degraded (negative = decreasing).
/// DegradationMinutes is how long until full degradation is reached.
/// </summary>
public sealed record TagSpec(
    string Name,
    double BaseValue,
    double NoiseAmplitude,
    double DegradationDelta = 0.0,
    double DegradationMinutes = 60.0)
{
    public double DegradedValue(double minutesElapsed)
    {
        if (DegradationDelta == 0.0) return BaseValue;
        var factor = Math.Min(1.0, minutesElapsed / DegradationMinutes);
        return BaseValue + DegradationDelta * factor;
    }
}

public sealed class AssetProfile
{
    public required string AssetId { get; init; }
    public required string AssetType { get; init; }
    public AssetHealthState HealthState { get; init; } = AssetHealthState.Healthy;
    public required IReadOnlyList<TagSpec> TagSpecs { get; init; }
}

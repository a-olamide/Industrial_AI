using System.Text.Json.Serialization;

namespace IndustrialTelemetry.Producer.Models;

public sealed record TelemetryEvent
{
    [JsonPropertyName("asset_id")]
    public required string AssetId { get; init; }

    [JsonPropertyName("asset_type")]
    public required string AssetType { get; init; }

    [JsonPropertyName("ts")]
    public required string Ts { get; init; }

    [JsonPropertyName("tag")]
    public required string Tag { get; init; }

    [JsonPropertyName("value")]
    public required double Value { get; init; }

    [JsonPropertyName("quality")]
    public required int Quality { get; init; }
}

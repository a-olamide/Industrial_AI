namespace IndustrialAnalytics.Contracts.DigitalTwins;

/// <summary>
/// INTEGRATION POINT — reserved for a future explanation service.
///
/// <para>
/// This is the evidence bundle a narration service (e.g. the Claude API)
/// will consume to turn ML output into maintenance guidance. It is
/// defined now so the seam is explicit, but <b>nothing calls an LLM in
/// this phase</b> — there is deliberately no HTTP client, no endpoint
/// and no service registration behind it.
/// </para>
///
/// <para>
/// Note what is and is not here. The explainer receives MODEL OUTPUT
/// (<see cref="Anomaly"/>, <see cref="Classification"/>), the inputs
/// that produced it (<see cref="Features"/>,
/// <see cref="OperatingContext"/>) and window provenance. It does NOT
/// receive demo ground truth: an explanation grounded in the answer key
/// would be worthless on a real machine, which is exactly the situation
/// the explainer exists for.
/// </para>
/// </summary>
public sealed record HealthExplanationContextDto(
    string AssetId,
    DateTime ObservedAtUtc,
    TwinWindowDto Window,
    TwinAnomalyDto Anomaly,
    TwinClassificationDto Classification,
    TwinFeaturesDto Features,
    TwinOperatingContextDto OperatingContext
);

/// <summary>
/// Builds the explanation evidence bundle from current twin state.
/// Pure projection: no I/O, no model call, no LLM.
/// </summary>
public static class HealthExplanationContextFactory
{
    /// <summary>
    /// Projects a <see cref="DigitalTwinStateDto"/> into the evidence a
    /// future explanation service needs, dropping demo ground truth.
    /// </summary>
    public static HealthExplanationContextDto FromTwinState(DigitalTwinStateDto state)
        => new(
            state.AssetId,
            state.LastUpdatedUtc,
            state.Window,
            state.Anomaly,
            state.Classification,
            state.Features,
            state.OperatingContext);
}

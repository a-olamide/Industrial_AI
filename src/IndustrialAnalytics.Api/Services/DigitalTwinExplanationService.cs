using System.Text.Json;
using IndustrialAnalytics.Contracts.DigitalTwins;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Options;

namespace IndustrialAnalytics.Api.Services
{
    /// <summary>
    /// Turns authoritative Digital Twin state into a human-readable
    /// maintenance explanation.
    ///
    /// <para>
    /// <b>ML predicts; Claude explains.</b> The trained Isolation Forest and
    /// Random Forest remain the only sources of the anomaly verdict and the
    /// fault class. This service reads their output server-side, hands
    /// Claude a narrow evidence bundle, and returns prose. It never asks
    /// Claude what the fault is, and the response envelope restates the ML
    /// verdict alongside the prose so a client can always see what the
    /// models actually said.
    /// </para>
    ///
    /// <para>
    /// The caller supplies only an asset id. Letting a client post arbitrary
    /// feature values would let the UI fabricate an ML result and have the
    /// server narrate it authoritatively.
    /// </para>
    /// </summary>
    public sealed class DigitalTwinExplanationService(
        IDigitalTwinQueryRepository repository,
        IStructuredLlmClient claude,
        IMemoryCache cache,
        IOptions<ClaudeOptions> options,
        ILogger<DigitalTwinExplanationService> logger)
    {
        private readonly ClaudeOptions _options = options.Value;

        private static readonly JsonSerializerOptions SerializerOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = false,
        };

        public const string SystemPrompt = """
            You explain machine-health results that were produced by trained
            machine-learning models monitoring industrial rotating equipment.
            You are not the detector and not the classifier. Your only job is
            to translate the supplied structured result into a concise
            maintenance explanation for an industrial dashboard.

            Hard constraints:
            - Do NOT change, second-guess or override the supplied anomaly
              result. If isAnomalous is false, the machine was not flagged.
            - Do NOT change the supplied predictedClass. If the classifier
              says BALL, your explanation is about BALL, even if the feature
              values suggest something else to you. Say what the model said.
            - Do NOT claim more certainty than the supplied confidence
              supports.
            - Do NOT invent sensor readings, measurements, thresholds, trends,
              part numbers, operating hours or maintenance history. Use only
              the numbers supplied.
            - Do NOT propose causes that the supplied evidence does not
              support.
            - Always distinguish a MODEL PREDICTION from a confirmed physical
              diagnosis. A prediction is a hypothesis to be verified by
              inspection.
            - Recommend inspection, measurement and monitoring. Do not present
              bearing replacement, teardown or other costly/destructive work
              as unquestionably necessary; frame it as conditional on what an
              inspection finds.
            - If confidence is low, or two or more class probabilities are
              close together, state the uncertainty explicitly and name the
              competing classes.
            - If the anomaly detector and the classifier disagree - the
              detector says normal while the classifier names a fault, or the
              detector flags an anomaly the classifier calls NORMAL - say so
              plainly and explain what the disagreement means. Never hide it.
            - If both models indicate normal operation, do not invent a fault.
              Say the current model outputs show no detected bearing fault and
              recommend continued routine monitoring.

            Vocabulary for this equipment: NORMAL means no bearing fault
            detected. INNER_RACE, BALL and OUTER_RACE are bearing fault
            locations. The anomaly score follows the convention supplied with
            it; a lower score is more anomalous.

            Be concise - this renders in a dashboard card. Summary at most
            three sentences; at most four evidence points; at most four
            recommended actions; each item one short sentence.
            """;

        /// <summary>
        /// Response schema. Claude is constrained to it, so the UI's fields
        /// are guaranteed rather than parsed hopefully.
        /// </summary>
        public static JsonElement ResponseSchema { get; } = JsonSerializer.Deserialize<JsonElement>("""
            {
              "type": "object",
              "additionalProperties": false,
              "required": ["summary", "likelyCondition", "evidence", "recommendedActions", "confidenceNote"],
              "properties": {
                "summary": {
                  "type": "string",
                  "description": "At most three sentences assessing the current machine state."
                },
                "likelyCondition": {
                  "type": "string",
                  "description": "The condition implied by the supplied model output, phrased as a model prediction rather than a confirmed diagnosis."
                },
                "evidence": {
                  "type": "array",
                  "maxItems": 4,
                  "items": { "type": "string" },
                  "description": "Specific supplied values that support the assessment."
                },
                "recommendedActions": {
                  "type": "array",
                  "maxItems": 4,
                  "items": { "type": "string" },
                  "description": "Inspection and monitoring steps, not assertions of required repair."
                },
                "confidenceNote": {
                  "type": "string",
                  "description": "How much weight to place on this, including any model disagreement or close class probabilities."
                }
              }
            }
            """);

        public async Task<MaintenanceExplanationResponseDto> ExplainAsync(
            string assetId, bool refresh, CancellationToken ct)
        {
            // Ground truth is excluded at the source: includeGroundTruth:false
            // means the demo columns never even leave the database on this
            // path, so they cannot reach the prompt by accident.
            var state = await repository.GetByAssetAsync(assetId, includeGroundTruth: false, ct)
                ?? throw new TwinNotFoundException($"No Digital Twin state for asset '{assetId}'.");

            var cacheKey = $"twin-explanation:{state.AssetId}:{state.Window.WindowEndSequence}";
            if (_options.CacheEnabled && !refresh &&
                cache.TryGetValue(cacheKey, out MaintenanceExplanationResponseDto? cached) &&
                cached is not null)
            {
                logger.LogInformation(
                    "Serving cached explanation for {AssetId} window {Window}.",
                    state.AssetId, state.Window.WindowEndSequence);
                return cached with { FromCache = true };
            }

            if (!claude.IsConfigured)
            {
                throw new ClaudeNotConfiguredException(
                    $"{ClaudeOptions.ApiKeyEnvironmentVariable} is not configured on the server.");
            }

            var request = await BuildRequestAsync(state, ct);
            var userJson = JsonSerializer.Serialize(request, SerializerOptions);

            var raw = await claude.CompleteStructuredAsync(
                SystemPrompt, userJson, ResponseSchema, ct);

            var explanation = ParseExplanation(raw);

            var response = new MaintenanceExplanationResponseDto(
                state.AssetId,
                state.Window.WindowEndSequence,
                DateTime.UtcNow,
                claude.ModelName,
                FromCache: false,
                new MlVerdictDto(
                    state.Anomaly.IsAnomalous,
                    state.Anomaly.Score,
                    state.Classification.PredictedClass,
                    state.Classification.Confidence),
                explanation);

            if (_options.CacheEnabled)
            {
                cache.Set(cacheKey, response, TimeSpan.FromMinutes(_options.CacheMinutes));
            }
            return response;
        }

        /// <summary>
        /// Builds the ONLY payload that reaches Claude.
        /// <para>
        /// <paramref name="state"/> is already loaded with
        /// <c>includeGroundTruth:false</c>, and this projection copies named
        /// fields rather than spreading the state object, so adding a demo
        /// column to the twin later cannot silently widen the prompt.
        /// </para>
        /// </summary>
        public async Task<ClaudeExplanationRequestDto> BuildRequestAsync(
            DigitalTwinStateDto state, CancellationToken ct)
        {
            TwinTrendDto? trend = null;
            try
            {
                var history = await repository.GetHistoryAsync(
                    state.AssetId, _options.TrendWindowCount, includeGroundTruth: false, ct);

                if (history.Count > 0)
                {
                    trend = new TwinTrendDto(
                        history.Count,
                        history.Count(h => h.IsAnomalous),
                        history.GroupBy(h => h.PredictedClass)
                               .ToDictionary(g => g.Key, g => g.Count()));
                }
            }
            catch (Exception ex)
            {
                // Trend is a nicety; its absence must not block an explanation.
                logger.LogWarning(ex, "Could not load trend for {AssetId}.", state.AssetId);
            }

            return new ClaudeExplanationRequestDto(
                state.AssetId,
                state.LastUpdatedUtc,
                state.Anomaly,
                state.Classification,
                state.Features,
                state.OperatingContext,
                trend);
        }

        /// <summary>
        /// Validates Claude's output. Malformed AI output must degrade, never
        /// break the Digital Twin page.
        /// </summary>
        public static MaintenanceExplanationDto ParseExplanation(string raw)
        {
            MaintenanceExplanationDto? parsed = null;
            try
            {
                parsed = JsonSerializer.Deserialize<MaintenanceExplanationDto>(
                    raw,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            }
            catch (JsonException)
            {
                // fall through to the guarded result below
            }

            if (parsed is null || string.IsNullOrWhiteSpace(parsed.Summary))
            {
                throw new ClaudeMalformedResponseException(
                    "Claude returned a response that did not match the explanation schema.");
            }

            return parsed with
            {
                LikelyCondition = string.IsNullOrWhiteSpace(parsed.LikelyCondition)
                    ? "Not stated"
                    : parsed.LikelyCondition,
                Evidence = parsed.Evidence ?? [],
                RecommendedActions = parsed.RecommendedActions ?? [],
                ConfidenceNote = parsed.ConfidenceNote ?? "",
            };
        }
    }

    public sealed class TwinNotFoundException(string message) : Exception(message);

    public sealed class ClaudeMalformedResponseException(string message) : Exception(message);
}

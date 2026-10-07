using System.Text.Json;
using IndustrialAnalytics.Api.Services;
using IndustrialAnalytics.Contracts.DigitalTwins;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace IndustrialAnalytics.Tests;

// ─────────────────────────────────────────────────────────────────────────
// Test doubles. Claude is ALWAYS mocked - no test in this file makes a paid
// or live API call. Live verification is a separate, explicit command
// documented in ml/streaming/README.md.
// ─────────────────────────────────────────────────────────────────────────

/// <summary>
/// Captures exactly what would have been sent to Claude, so tests can assert
/// on the real serialized payload rather than on an abstraction of it.
/// </summary>
internal sealed class FakeClaudeClient : IStructuredLlmClient
{
    public string? CapturedSystemPrompt { get; private set; }
    public string? CapturedUserJson { get; private set; }
    public JsonElement CapturedSchema { get; private set; }
    public int CallCount { get; private set; }

    public string Response { get; set; } = ValidResponse;
    public Exception? ThrowOnCall { get; set; }
    public bool IsConfigured { get; set; } = true;
    public string ModelName => "claude-haiku-4-5";

    public const string ValidResponse = """
        {
          "summary": "The anomaly detector flagged this window and the classifier associates it with an outer-race pattern.",
          "likelyCondition": "Model prediction: OUTER_RACE bearing fault (not a confirmed diagnosis)",
          "evidence": ["Anomaly score -0.7382 is below the -0.6153 threshold."],
          "recommendedActions": ["Schedule a vibration inspection of the drive-end bearing."],
          "confidenceNote": "Classifier confidence is 81%; verify by inspection."
        }
        """;

    public Task<string> CompleteJsonAsync(string systemPrompt, string userJson, CancellationToken ct)
        => CompleteStructuredAsync(systemPrompt, userJson, default, ct);

    public Task<string> CompleteStructuredAsync(
        string systemPrompt, string userJson, JsonElement jsonSchema, CancellationToken ct)
    {
        CallCount++;
        CapturedSystemPrompt = systemPrompt;
        CapturedUserJson = userJson;
        CapturedSchema = jsonSchema;
        if (ThrowOnCall is not null) throw ThrowOnCall;
        return Task.FromResult(Response);
    }
}

/// <summary>
/// Stands in for the SQL repository.
/// <para>
/// Critically, it models the real contract: <c>includeGroundTruth:false</c>
/// strips the demo block. That is what lets a test prove the service asks
/// for state WITHOUT ground truth rather than merely declining to copy it.
/// </para>
/// </summary>
internal sealed class FakeTwinRepository(DigitalTwinStateDto? state) : IDigitalTwinQueryRepository
{
    public bool? LastIncludeGroundTruth { get; private set; }
    public bool? LastHistoryIncludeGroundTruth { get; private set; }
    public List<DigitalTwinHistoryPointDto> History { get; set; } = [];

    public Task<IReadOnlyList<DigitalTwinStateDto>> GetAllAsync(bool includeGroundTruth, CancellationToken ct)
        => Task.FromResult<IReadOnlyList<DigitalTwinStateDto>>(
            state is null ? [] : [Project(state, includeGroundTruth)]);

    public Task<DigitalTwinStateDto?> GetByAssetAsync(string assetId, bool includeGroundTruth, CancellationToken ct)
    {
        LastIncludeGroundTruth = includeGroundTruth;
        if (state is null || state.AssetId != assetId) return Task.FromResult<DigitalTwinStateDto?>(null);
        return Task.FromResult<DigitalTwinStateDto?>(Project(state, includeGroundTruth));
    }

    public Task<IReadOnlyList<DigitalTwinHistoryPointDto>> GetHistoryAsync(
        string assetId, int take, bool includeGroundTruth, CancellationToken ct)
    {
        LastHistoryIncludeGroundTruth = includeGroundTruth;
        var rows = History.Take(take).Select(h => includeGroundTruth ? h : h with { DemoFaultClass = null });
        return Task.FromResult<IReadOnlyList<DigitalTwinHistoryPointDto>>(rows.ToList());
    }

    private static DigitalTwinStateDto Project(DigitalTwinStateDto s, bool includeGroundTruth)
        => includeGroundTruth ? s : s with { DemoGroundTruth = null };
}

internal static class Fixtures
{
    /// <summary>
    /// The difficult demo case: the classifier says BALL while the demo
    /// ground truth records OUTER_RACE (OR014@6_3).
    /// </summary>
    public static DigitalTwinStateDto Motor003BallPrediction() => new(
        "MOTOR_003",
        new DateTime(2026, 10, 7, 12, 0, 0, DateTimeKind.Utc),
        new TwinWindowDto(118784, 120831, 2048),
        new TwinAnomalyDto(true, -0.7653, -0.6153),
        new TwinClassificationDto("BALL", 0.68, new Dictionary<string, double>
        {
            ["BALL"] = 0.68, ["OUTER_RACE"] = 0.32, ["INNER_RACE"] = 0.0, ["NORMAL"] = 0.0,
        }),
        new TwinFeaturesDto(0.0955, 0.0954, 0.4175, 0.8123, 0.469, 0.0123, 4.370),
        new TwinOperatingContextDto(3.0, 1723.0),
        new TwinGroundTruthDto("OR014@6_3", "OUTER_RACE", 0.014));

    public static DigitalTwinStateDto NormalState() => new(
        "MOTOR_001",
        new DateTime(2026, 10, 7, 12, 0, 0, DateTimeKind.Utc),
        new TwinWindowDto(483328, 485375, 2048),
        new TwinAnomalyDto(false, -0.4242, -0.6153),
        new TwinClassificationDto("NORMAL", 1.0, new Dictionary<string, double>
        {
            ["NORMAL"] = 1.0, ["BALL"] = 0.0, ["INNER_RACE"] = 0.0, ["OUTER_RACE"] = 0.0,
        }),
        new TwinFeaturesDto(0.0670, 0.0651, 0.2232, 0.4387, -0.0233, -0.0531, 3.3308),
        new TwinOperatingContextDto(3.0, 1725.0),
        new TwinGroundTruthDto("Normal_3", "NORMAL", null));

    public static DigitalTwinStateDto AnomalousHighConfidence() => new(
        "MOTOR_002",
        new DateTime(2026, 10, 7, 12, 0, 0, DateTimeKind.Utc),
        new TwinWindowDto(118784, 120831, 2048),
        new TwinAnomalyDto(true, -0.7782, -0.6153),
        new TwinClassificationDto("OUTER_RACE", 1.0, new Dictionary<string, double>
        {
            ["OUTER_RACE"] = 1.0, ["BALL"] = 0.0, ["INNER_RACE"] = 0.0, ["NORMAL"] = 0.0,
        }),
        new TwinFeaturesDto(0.5433, 0.5432, 4.8211, 9.2645, 17.6976, -0.0743, 8.8744),
        new TwinOperatingContextDto(3.0, 1721.0),
        new TwinGroundTruthDto("OR021@6_3", "OUTER_RACE", 0.021));

    /// <summary>Detector says normal, classifier names a fault.</summary>
    public static DigitalTwinStateDto Disagreement() => new(
        "MOTOR_004",
        new DateTime(2026, 10, 7, 12, 0, 0, DateTimeKind.Utc),
        new TwinWindowDto(0, 2047, 2048),
        new TwinAnomalyDto(false, -0.4100, -0.6153),
        new TwinClassificationDto("INNER_RACE", 0.55, new Dictionary<string, double>
        {
            ["INNER_RACE"] = 0.55, ["BALL"] = 0.40, ["NORMAL"] = 0.05, ["OUTER_RACE"] = 0.0,
        }),
        new TwinFeaturesDto(0.1, 0.1, 0.4, 0.8, 1.2, 0.01, 4.0),
        new TwinOperatingContextDto(3.0, 1725.0),
        null);

    public static DigitalTwinExplanationService Service(
        FakeTwinRepository repo, FakeClaudeClient claude, ClaudeOptions? options = null)
        => new(
            repo,
            claude,
            new MemoryCache(new MemoryCacheOptions()),
            Options.Create(options ?? new ClaudeOptions { ApiKey = "test-key-not-real" }),
            NullLogger<DigitalTwinExplanationService>.Instance);
}

// ─────────────────────────────────────────────────────────────────────────
// What reaches Claude
// ─────────────────────────────────────────────────────────────────────────

public class ClaudeRequestContentTests
{
    private static async Task<string> CapturePayloadAsync(DigitalTwinStateDto state)
    {
        var repo = new FakeTwinRepository(state);
        var claude = new FakeClaudeClient();
        await Fixtures.Service(repo, claude).ExplainAsync(state.AssetId, refresh: false, default);
        Assert.NotNull(claude.CapturedUserJson);
        return claude.CapturedUserJson!;
    }

    [Fact]
    public async Task Request_contains_the_expected_ml_fields()
    {
        var json = await CapturePayloadAsync(Fixtures.Motor003BallPrediction());
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        Assert.Equal("MOTOR_003", root.GetProperty("assetId").GetString());
        Assert.True(root.GetProperty("anomaly").GetProperty("isAnomalous").GetBoolean());
        Assert.Equal(-0.7653, root.GetProperty("anomaly").GetProperty("score").GetDouble(), 4);
        Assert.Equal("BALL", root.GetProperty("classification").GetProperty("predictedClass").GetString());
        Assert.Equal(0.68, root.GetProperty("classification").GetProperty("confidence").GetDouble(), 4);
        Assert.True(root.GetProperty("classification").TryGetProperty("probabilities", out _));
        Assert.Equal(3.0, root.GetProperty("operatingContext").GetProperty("motorLoadHp").GetDouble());
        Assert.Equal(1723.0, root.GetProperty("operatingContext").GetProperty("rotationalSpeedRpm").GetDouble());
    }

    [Fact]
    public async Task Request_contains_all_seven_engineered_features()
    {
        var json = await CapturePayloadAsync(Fixtures.Motor003BallPrediction());
        using var doc = JsonDocument.Parse(json);
        var features = doc.RootElement.GetProperty("features");

        foreach (var name in new[]
        {
            "vibrationRms", "vibrationStd", "vibrationPeak", "vibrationPeakToPeak",
            "vibrationKurtosis", "vibrationSkewness", "crestFactor",
        })
        {
            Assert.True(features.TryGetProperty(name, out _), $"missing feature {name}");
        }
        Assert.Equal(7, features.EnumerateObject().Count());
    }

    [Fact]
    public async Task Request_contains_no_demo_ground_truth()
    {
        var json = await CapturePayloadAsync(Fixtures.Motor003BallPrediction());

        Assert.DoesNotContain("demoGroundTruth", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("groundTruth", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("faultSeverityIn", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("recordingId", json, StringComparison.OrdinalIgnoreCase);
        // The literal recording name and severity value must be absent too.
        Assert.DoesNotContain("OR014", json, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("0.014", json, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Request_contains_no_raw_vibration_sample_array()
    {
        var json = await CapturePayloadAsync(Fixtures.Motor003BallPrediction());
        using var doc = JsonDocument.Parse(json);

        // Only the probabilities map and the trend map are objects; no node
        // anywhere may be an array of readings.
        static void AssertNoArrays(JsonElement node, string path)
        {
            switch (node.ValueKind)
            {
                case JsonValueKind.Array:
                    Assert.Fail($"unexpected array at {path}");
                    break;
                case JsonValueKind.Object:
                    foreach (var p in node.EnumerateObject())
                        AssertNoArrays(p.Value, $"{path}.{p.Name}");
                    break;
            }
        }
        AssertNoArrays(doc.RootElement, "$");
        Assert.True(json.Length < 2000, $"payload unexpectedly large ({json.Length} chars)");
    }

    [Fact]
    public async Task Service_loads_twin_state_without_ground_truth()
    {
        var repo = new FakeTwinRepository(Fixtures.Motor003BallPrediction());
        var claude = new FakeClaudeClient();
        await Fixtures.Service(repo, claude).ExplainAsync("MOTOR_003", false, default);

        // The demo columns never leave the database on this path.
        Assert.False(repo.LastIncludeGroundTruth);
        Assert.False(repo.LastHistoryIncludeGroundTruth);
    }

    [Fact]
    public async Task Trend_is_included_and_carries_only_counts()
    {
        var state = Fixtures.Motor003BallPrediction();
        var repo = new FakeTwinRepository(state)
        {
            History =
            [
                new(1, DateTime.UtcNow, 0, 2047, true, -0.77, "OUTER_RACE", 0.9, 0.09, 0.4, "OUTER_RACE"),
                new(2, DateTime.UtcNow, 2048, 4095, true, -0.78, "BALL", 0.8, 0.09, 0.4, "OUTER_RACE"),
            ],
        };
        var claude = new FakeClaudeClient();
        await Fixtures.Service(repo, claude).ExplainAsync("MOTOR_003", false, default);

        using var doc = JsonDocument.Parse(claude.CapturedUserJson!);
        var trend = doc.RootElement.GetProperty("recentTrend");
        Assert.Equal(2, trend.GetProperty("windowsConsidered").GetInt32());
        Assert.Equal(2, trend.GetProperty("windowsFlaggedAnomalous").GetInt32());
        Assert.Equal(1, trend.GetProperty("predictedClassCounts").GetProperty("BALL").GetInt32());
        // Even inside the trend, no ground truth.
        Assert.DoesNotContain("OR014", claude.CapturedUserJson!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Api_key_is_never_part_of_the_payload_or_the_response()
    {
        // Deliberately NOT shaped like a real key, so repository secret
        // scanners never flag this fixture.
        const string secret = "FAKE-TEST-CREDENTIAL-MUST-NEVER-LEAK";
        var repo = new FakeTwinRepository(Fixtures.NormalState());
        var claude = new FakeClaudeClient();
        var service = Fixtures.Service(repo, claude, new ClaudeOptions { ApiKey = secret });

        var response = await service.ExplainAsync("MOTOR_001", false, default);
        var serialized = JsonSerializer.Serialize(response);

        Assert.DoesNotContain(secret, claude.CapturedUserJson!, StringComparison.Ordinal);
        Assert.DoesNotContain(secret, claude.CapturedSystemPrompt!, StringComparison.Ordinal);
        Assert.DoesNotContain(secret, serialized, StringComparison.Ordinal);
        Assert.DoesNotContain("apiKey", serialized, StringComparison.OrdinalIgnoreCase);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MOTOR_003 — the required demonstration case
// ─────────────────────────────────────────────────────────────────────────

public class Motor003DisagreementTests
{
    [Fact]
    public async Task Ball_prediction_is_sent_as_BALL_even_though_ground_truth_says_OUTER_RACE()
    {
        var state = Fixtures.Motor003BallPrediction();
        Assert.Equal("OUTER_RACE", state.DemoGroundTruth!.FaultClass); // the fixture really does disagree

        var repo = new FakeTwinRepository(state);
        var claude = new FakeClaudeClient();
        var response = await Fixtures.Service(repo, claude).ExplainAsync("MOTOR_003", false, default);

        using var doc = JsonDocument.Parse(claude.CapturedUserJson!);
        var classification = doc.RootElement.GetProperty("classification");
        Assert.Equal("BALL", classification.GetProperty("predictedClass").GetString());

        // OUTER_RACE DOES legitimately appear - but only as a 32% entry in the
        // classifier's own probability map, which Claude needs in order to
        // state the uncertainty. What must be absent is any assertion that
        // OUTER_RACE is the true condition.
        Assert.Equal(
            0.32,
            classification.GetProperty("probabilities").GetProperty("OUTER_RACE").GetDouble(),
            4);
        Assert.DoesNotContain("demoGroundTruth", claude.CapturedUserJson!, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("faultClass", claude.CapturedUserJson!, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("OR014", claude.CapturedUserJson!, StringComparison.OrdinalIgnoreCase);

        // Every occurrence of the string is inside the probability map.
        var outsideProbabilities = claude.CapturedUserJson!.Replace(
            classification.GetProperty("probabilities").GetRawText(), "");
        Assert.DoesNotContain("OUTER_RACE", outsideProbabilities, StringComparison.Ordinal);

        // And the echoed ML verdict still reports what the classifier said.
        Assert.Equal("BALL", response.MlVerdict.PredictedClass);
    }

    [Fact]
    public async Task Close_probabilities_are_supplied_so_uncertainty_can_be_stated()
    {
        var repo = new FakeTwinRepository(Fixtures.Motor003BallPrediction());
        var claude = new FakeClaudeClient();
        await Fixtures.Service(repo, claude).ExplainAsync("MOTOR_003", false, default);

        using var doc = JsonDocument.Parse(claude.CapturedUserJson!);
        var probabilities = doc.RootElement.GetProperty("classification").GetProperty("probabilities");
        Assert.Equal(0.68, probabilities.GetProperty("BALL").GetDouble(), 4);
        Assert.Equal(0.32, probabilities.GetProperty("OUTER_RACE").GetDouble(), 4);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Prompt constraints (Tasks 4, 9, 10)
// ─────────────────────────────────────────────────────────────────────────

public class SystemPromptTests
{
    private static readonly string Prompt = DigitalTwinExplanationService.SystemPrompt;

    [Theory]
    [InlineData("Do NOT change")]            // no overriding model output
    [InlineData("predictedClass")]           // classifier result is fixed
    [InlineData("Do NOT invent sensor")]     // no fabricated readings
    [InlineData("maintenance history")]      // no fabricated history
    [InlineData("MODEL PREDICTION")]         // prediction != diagnosis
    [InlineData("inspection")]               // recommend inspection
    [InlineData("confidence is low")]        // uncertainty
    [InlineData("disagree")]                 // disagreement is explained
    [InlineData("do not invent a fault")]    // normal case
    public void Prompt_states_each_required_constraint(string fragment)
        => Assert.Contains(fragment, Prompt, StringComparison.OrdinalIgnoreCase);

    [Fact]
    public void Prompt_tells_the_model_it_is_not_the_detector_or_classifier()
    {
        Assert.Contains("not the detector", Prompt, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("not the classifier", Prompt, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Prompt_never_mentions_ground_truth_or_the_dataset()
    {
        Assert.DoesNotContain("ground truth", Prompt, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("CWRU", Prompt, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("recordingId", Prompt, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Response_schema_requires_every_ui_field()
    {
        var schema = DigitalTwinExplanationService.ResponseSchema;
        var required = schema.GetProperty("required").EnumerateArray()
            .Select(e => e.GetString()).ToHashSet();

        Assert.Equal(
            new HashSet<string?> { "summary", "likelyCondition", "evidence", "recommendedActions", "confidenceNote" },
            required);
        Assert.False(schema.GetProperty("additionalProperties").GetBoolean());
    }

    [Fact]
    public async Task Schema_is_actually_sent_with_the_request()
    {
        var repo = new FakeTwinRepository(Fixtures.NormalState());
        var claude = new FakeClaudeClient();
        await Fixtures.Service(repo, claude).ExplainAsync("MOTOR_001", false, default);

        Assert.Equal(JsonValueKind.Object, claude.CapturedSchema.ValueKind);
        Assert.True(claude.CapturedSchema.TryGetProperty("required", out _));
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario coverage (Task 10 A-D)
// ─────────────────────────────────────────────────────────────────────────

public class ScenarioTests
{
    private static async Task<JsonElement> PayloadFor(DigitalTwinStateDto state)
    {
        var repo = new FakeTwinRepository(state);
        var claude = new FakeClaudeClient();
        await Fixtures.Service(repo, claude).ExplainAsync(state.AssetId, false, default);
        return JsonDocument.Parse(claude.CapturedUserJson!).RootElement.Clone();
    }

    [Fact] // A: normal + normal
    public async Task Normal_state_is_conveyed_as_not_anomalous_and_NORMAL()
    {
        var payload = await PayloadFor(Fixtures.NormalState());
        Assert.False(payload.GetProperty("anomaly").GetProperty("isAnomalous").GetBoolean());
        Assert.Equal("NORMAL", payload.GetProperty("classification").GetProperty("predictedClass").GetString());
    }

    [Fact] // B: anomalous + high confidence
    public async Task Anomalous_high_confidence_is_conveyed_intact()
    {
        var payload = await PayloadFor(Fixtures.AnomalousHighConfidence());
        Assert.True(payload.GetProperty("anomaly").GetProperty("isAnomalous").GetBoolean());
        Assert.Equal("OUTER_RACE", payload.GetProperty("classification").GetProperty("predictedClass").GetString());
        Assert.Equal(1.0, payload.GetProperty("classification").GetProperty("confidence").GetDouble(), 4);
    }

    [Fact] // C: anomalous + low confidence / close probabilities
    public async Task Anomalous_low_confidence_supplies_the_competing_classes()
    {
        var payload = await PayloadFor(Fixtures.Motor003BallPrediction());
        Assert.True(payload.GetProperty("anomaly").GetProperty("isAnomalous").GetBoolean());
        var probabilities = payload.GetProperty("classification").GetProperty("probabilities");
        var ordered = probabilities.EnumerateObject()
            .Select(p => p.Value.GetDouble()).OrderByDescending(v => v).ToList();
        Assert.True(ordered[0] - ordered[1] < 0.5, "fixture should have close probabilities");
    }

    [Fact] // D: detector normal, classifier predicts a fault
    public async Task Model_disagreement_is_visible_in_the_payload()
    {
        var payload = await PayloadFor(Fixtures.Disagreement());
        var anomalous = payload.GetProperty("anomaly").GetProperty("isAnomalous").GetBoolean();
        var predicted = payload.GetProperty("classification").GetProperty("predictedClass").GetString();

        Assert.False(anomalous);
        Assert.NotEqual("NORMAL", predicted);
        // Both halves are present, so the prompt's disagreement rule can fire.
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Failure handling
// ─────────────────────────────────────────────────────────────────────────

public class ErrorHandlingTests
{
    [Fact]
    public async Task Missing_asset_throws_TwinNotFound()
    {
        var service = Fixtures.Service(new FakeTwinRepository(null), new FakeClaudeClient());
        await Assert.ThrowsAsync<TwinNotFoundException>(
            () => service.ExplainAsync("NO_SUCH_ASSET", false, default));
    }

    [Fact]
    public async Task Missing_api_key_throws_ClaudeNotConfigured()
    {
        var claude = new FakeClaudeClient { IsConfigured = false };
        var service = Fixtures.Service(
            new FakeTwinRepository(Fixtures.NormalState()), claude,
            new ClaudeOptions { ApiKey = null });

        await Assert.ThrowsAsync<ClaudeNotConfiguredException>(
            () => service.ExplainAsync("MOTOR_001", false, default));
        Assert.Equal(0, claude.CallCount); // and no request was attempted
    }

    [Fact]
    public async Task Unconfigured_error_names_the_variable_but_no_secret()
    {
        var service = Fixtures.Service(
            new FakeTwinRepository(Fixtures.NormalState()),
            new FakeClaudeClient { IsConfigured = false },
            new ClaudeOptions { ApiKey = null });

        var ex = await Assert.ThrowsAsync<ClaudeNotConfiguredException>(
            () => service.ExplainAsync("MOTOR_001", false, default));
        Assert.Contains("ANTHROPIC_API_KEY", ex.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("sk-ant", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("this is not json at all")]
    [InlineData("{\"unexpected\": \"shape\"}")]
    [InlineData("{\"summary\": \"\"}")]
    [InlineData("")]
    public void Malformed_claude_output_raises_a_typed_error(string raw)
        => Assert.Throws<ClaudeMalformedResponseException>(
            () => DigitalTwinExplanationService.ParseExplanation(raw));

    [Fact]
    public async Task Malformed_output_surfaces_as_a_typed_error_from_the_service()
    {
        var claude = new FakeClaudeClient { Response = "{\"nope\": 1}" };
        var service = Fixtures.Service(new FakeTwinRepository(Fixtures.NormalState()), claude);

        await Assert.ThrowsAsync<ClaudeMalformedResponseException>(
            () => service.ExplainAsync("MOTOR_001", false, default));
    }

    [Fact]
    public async Task Upstream_failure_propagates_as_ClaudeUnavailable()
    {
        var claude = new FakeClaudeClient { ThrowOnCall = new ClaudeUnavailableException("timeout") };
        var service = Fixtures.Service(new FakeTwinRepository(Fixtures.NormalState()), claude);

        await Assert.ThrowsAsync<ClaudeUnavailableException>(
            () => service.ExplainAsync("MOTOR_001", false, default));
    }

    [Fact]
    public void Partial_but_valid_output_is_filled_rather_than_rejected()
    {
        var parsed = DigitalTwinExplanationService.ParseExplanation(
            "{\"summary\":\"Something happened.\"}");

        Assert.Equal("Something happened.", parsed.Summary);
        Assert.Equal("Not stated", parsed.LikelyCondition);
        Assert.Empty(parsed.Evidence);
        Assert.Empty(parsed.RecommendedActions);
    }

    [Fact]
    public void Valid_output_round_trips()
    {
        var parsed = DigitalTwinExplanationService.ParseExplanation(FakeClaudeClient.ValidResponse);
        Assert.Contains("anomaly detector", parsed.Summary, StringComparison.OrdinalIgnoreCase);
        Assert.Single(parsed.Evidence);
        Assert.Single(parsed.RecommendedActions);
        Assert.Contains("81%", parsed.ConfidenceNote, StringComparison.Ordinal);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Cost control
// ─────────────────────────────────────────────────────────────────────────

public class CostControlTests
{
    [Fact]
    public async Task Repeated_request_for_the_same_window_does_not_call_claude_again()
    {
        var repo = new FakeTwinRepository(Fixtures.NormalState());
        var claude = new FakeClaudeClient();
        var service = Fixtures.Service(repo, claude);

        var first = await service.ExplainAsync("MOTOR_001", false, default);
        var second = await service.ExplainAsync("MOTOR_001", false, default);

        Assert.Equal(1, claude.CallCount);
        Assert.False(first.FromCache);
        Assert.True(second.FromCache);
    }

    [Fact]
    public async Task Refresh_forces_a_new_call()
    {
        var repo = new FakeTwinRepository(Fixtures.NormalState());
        var claude = new FakeClaudeClient();
        var service = Fixtures.Service(repo, claude);

        await service.ExplainAsync("MOTOR_001", false, default);
        await service.ExplainAsync("MOTOR_001", refresh: true, default);

        Assert.Equal(2, claude.CallCount);
    }

    [Fact]
    public async Task A_new_window_is_explained_separately()
    {
        var state = Fixtures.NormalState();
        var repo = new FakeTwinRepository(state);
        var claude = new FakeClaudeClient();
        var service = Fixtures.Service(repo, claude);
        await service.ExplainAsync("MOTOR_001", false, default);

        // Same asset, next window -> different cache key.
        var advanced = state with { Window = new TwinWindowDto(485376, 487423, 2048) };
        var service2 = Fixtures.Service(new FakeTwinRepository(advanced), claude);
        await service2.ExplainAsync("MOTOR_001", false, default);

        Assert.Equal(2, claude.CallCount);
    }

    [Fact]
    public void Defaults_are_cost_conscious()
    {
        var options = new ClaudeOptions();
        Assert.Equal("claude-haiku-4-5", options.Model);
        Assert.True(options.MaxTokens <= 1024, "output cap should be small for a dashboard card");
        Assert.True(options.CacheEnabled);
    }
}

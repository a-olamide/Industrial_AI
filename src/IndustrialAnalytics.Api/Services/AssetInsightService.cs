using IndustrialAnalytics.Contracts.Insights;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;
using System.Text.Json;

namespace IndustrialAnalytics.Api.Services
{
    public sealed class AssetInsightService(
    ILlmClient llm,
    IAssetQueryRepository assets,
    IRiskQueryRepository riskRepo,
    IAnomalyQueryRepository anomalyRepo,
    IRecommendationQueryRepository recRepo)
    {
        public async Task<AssetInsightDto> BuildInsightAsync(string assetId, DateTime fromUtc, DateTime toUtc, int take, CancellationToken ct)
        {
            var summary = await assets.GetAssetSummaryAsync(assetId, ct);
            var risk = await riskRepo.GetRiskSeriesAsync(assetId, fromUtc, toUtc, stepMinutes: 1, ct);
            var anomalies = await anomalyRepo.GetForAssetAsync(assetId, fromUtc, toUtc, take: 200, ct);
            var recs = await recRepo.GetForAssetAsync(assetId, status: "OPEN", take: take, ct);

            var latest = risk.Points.LastOrDefault();
            var first = risk.Points.FirstOrDefault();

            string trend = "unknown";
            if (first is not null && latest is not null)
            {
                var delta = latest.Score - first.Score;
                trend = delta switch
                {
                    > 10 => "rising",
                    < -10 => "falling",
                    _ => "flat"
                };
            }

            var facts = new
            {
                assetId,
                windowUtc = new { fromUtc, toUtc },
                risk = new
                {
                    latest = latest,
                    trend,
                    min = risk.Points.Count == 0 ? (double?)null : risk.Points.Min(p => p.Score),
                    max = risk.Points.Count == 0 ? (double?)null : risk.Points.Max(p => p.Score),
                    avg = risk.Points.Count == 0 ? (double?)null : risk.Points.Average(p => p.Score)
                },
                anomalies = anomalies
                    .OrderByDescending(a => a.MinuteTs)
                    .Take(20)
                    .Select(a => new { a.MinuteTs, a.AnomalyType, a.Signal, a.Score, a.Severity, a.Reason })
                    .ToList(),
                recommendations = recs
                    .Take(10)
                    .Select(r => new { r.RecommendationId, r.RecType, r.Priority, r.Title, r.Description, r.Confidence, r.Drivers, r.Evidence })
                    .ToList()
            };

            //            var systemPrompt =
            //    """
            //You are an industrial reliability assistant for rotating equipment.
            //Rules:
            //- Use ONLY the provided facts.
            //- Do NOT invent sensor values, thresholds, or history.
            //- If data is missing, say so.
            //- Return STRICT JSON matching the required schema.
            //Required JSON schema:
            //{
            //  "headline": string,
            //  "summary": string,
            //  "what_changed": string[],
            //  "likely_causes": string[],
            //  "recommended_next_steps": string[],
            //  "confidence": number,
            //  "evidence_used": string[],
            //  "disclaimer": string
            //}
            //Keep it concise and CEO-friendly.
            //""";
            var systemPrompt =
                            """
                You are an industrial reliability assistant.

                Return ONLY a JSON object. No prose. No markdown. No extra keys.
                Use ONLY the facts provided.

                Do NOT include any fields other than the schema keys.
                Do NOT output "recommendations". Use "recommended_next_steps" (string list) instead.

                Output MUST contain EXACT keys:
                headline, summary, what_changed, likely_causes, recommended_next_steps, confidence, evidence_used, disclaimer

                Rules:
                - Arrays must be [] (never null)
                - Each array must contain at most 5 items
                - confidence must be a number 0..1
                - If you are uncertain, say so in summary and lower confidence
                - Keep it short and CEO-friendly
                """;

            var userJson = JsonSerializer.Serialize(facts);

            var insightJson = await llm.CompleteJsonAsync(systemPrompt, userJson, ct);

            // parse into DTO (safe fallback)
            //try
            //{
            //    var dto = JsonSerializer.Deserialize<AssetInsightDto>(insightJson,
            //        new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

            //    return dto ?? new AssetInsightDto(
            //        "Insight unavailable",
            //        "The insight engine returned an empty response.",
            //        Array.Empty<string>(),
            //        Array.Empty<string>(),
            //        Array.Empty<string>(),
            //        0.0,
            //        Array.Empty<string>(),
            //        "Generated text may be imperfect; verify against underlying signals."
            //    );
            //}
            //catch
            //{
            //    return new AssetInsightDto(
            //        "Insight unavailable",
            //        "The insight engine returned a non-JSON response.",
            //        Array.Empty<string>(),
            //        Array.Empty<string>(),
            //        Array.Empty<string>(),
            //        0.0,
            //        new[] { "llm_parse_error" },
            //        "Generated text may be imperfect; verify against underlying signals."
            //    );
            //}
            try
            {
                // If model returns extra text, extract JSON object first (optional but useful)
                insightJson = ExtractJsonObject(insightJson);

                return NormalizeInsight(insightJson);
            }
            catch
            {
                return new AssetInsightDto(
                    "Insight unavailable",
                    "The insight engine returned invalid JSON.",
                    Array.Empty<string>(),
                    Array.Empty<string>(),
                    Array.Empty<string>(),
                    0.0,
                    new[] { "llm_parse_error" },
                    "Generated text may be imperfect; verify against underlying signals."
                );
            }
        }
        private static string ExtractJsonObject(string s)
        {
            var start = s.IndexOf('{');
            var end = s.LastIndexOf('}');
            if (start >= 0 && end > start) return s.Substring(start, end - start + 1);
            return s;
        }
        private static AssetInsightDto NormalizeInsight(string rawJson)
        {
            using var doc = JsonDocument.Parse(rawJson);
            var root = doc.RootElement;

            string GetString(string name, string fallback = "") =>
                root.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String
                    ? v.GetString() ?? fallback
                    : fallback;

            double GetDouble(string name, double fallback = 0.0) =>
                root.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number
                    ? v.GetDouble()
                    : fallback;

            IReadOnlyList<string> GetStringArray(string name)
            {
                if (!root.TryGetProperty(name, out var v) || v.ValueKind != JsonValueKind.Array)
                    return Array.Empty<string>();

                return v.EnumerateArray()
                    .Where(x => x.ValueKind == JsonValueKind.String)
                    .Select(x => x.GetString() ?? "")
                    .Where(s => !string.IsNullOrWhiteSpace(s))
                    .ToList();
            }

            // If the model incorrectly returns "recommendations" (objects), extract titles as next steps
            IReadOnlyList<string> GetRecommendedNextSteps()
            {
                if (root.TryGetProperty("recommended_next_steps", out var good) && good.ValueKind == JsonValueKind.Array)
                    return GetStringArray("recommended_next_steps");

                if (root.TryGetProperty("recommendations", out var recs) && recs.ValueKind == JsonValueKind.Array)
                {
                    return recs.EnumerateArray()
                        .Select(r =>
                        {
                            if (r.ValueKind != JsonValueKind.Object) return null;
                            if (r.TryGetProperty("Title", out var t) && t.ValueKind == JsonValueKind.String) return t.GetString();
                            if (r.TryGetProperty("title", out var t2) && t2.ValueKind == JsonValueKind.String) return t2.GetString();
                            return null;
                        })
                        .Where(s => !string.IsNullOrWhiteSpace(s))
                        .Cast<string>()
                        .ToList();
                }

                return Array.Empty<string>();
            }

            return new AssetInsightDto(
                Headline: GetString("headline", "AI Insight"),
                Summary: GetString("summary", "No summary returned."),
                WhatChanged: GetStringArray("what_changed"),
                LikelyCauses: GetStringArray("likely_causes"),
                RecommendedNextSteps: GetRecommendedNextSteps(),
                Confidence: GetDouble("confidence", 0.0),
                EvidenceUsed: GetStringArray("evidence_used"),
                Disclaimer: GetString("disclaimer", "Generated output may be imperfect; verify against source data.")
            );
        }

    }
}

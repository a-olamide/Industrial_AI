using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Insights
{
    public sealed record AssetInsightDto(
    [property: JsonPropertyName("headline")] string Headline,
    [property: JsonPropertyName("summary")] string Summary,
    [property: JsonPropertyName("what_changed")] IReadOnlyList<string> WhatChanged,
    [property: JsonPropertyName("likely_causes")] IReadOnlyList<string> LikelyCauses,
    [property: JsonPropertyName("recommended_next_steps")] IReadOnlyList<string> RecommendedNextSteps,
    [property: JsonPropertyName("confidence")] double Confidence,
    [property: JsonPropertyName("evidence_used")] IReadOnlyList<string> EvidenceUsed,
    [property: JsonPropertyName("disclaimer")] string Disclaimer
);
}

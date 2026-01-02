using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Recommendations
{
    public sealed record RecommendationDto(
        long RecommendationId,
        string AssetId,
        DateTime AsOfMinuteTs,
        string RecType,
        int Priority,
        string Title,
        string Description,
        string Status,
        double Confidence,
        string? Drivers,
        string? Evidence,
        DateTime CreatedAt,
        DateTime UpdatedAt,
        DateTime? AckUntil
    );
}

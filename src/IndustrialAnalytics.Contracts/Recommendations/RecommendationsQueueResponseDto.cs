using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Recommendations
{
    public sealed record RecommendationsQueueResponseDto(
        string Status,
        int Take,
        string? AssetId,
        IReadOnlyList<RecommendationDto> Items
    );
}

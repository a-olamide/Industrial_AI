using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Assets
{
    public sealed record AssetRecommendationsResponseDto(
        string AssetId,
        IReadOnlyList<IndustrialAnalytics.Contracts.Recommendations.RecommendationDto> Items
    );
}

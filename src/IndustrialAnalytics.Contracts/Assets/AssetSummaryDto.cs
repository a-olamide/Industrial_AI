using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Assets
{
    public sealed record AssetSummaryDto(
        string AssetId,
        DateTime AsOfMinuteTs,
        RiskSummaryDto Risk,
        RecommendationCountsDto Recommendations
    );
}

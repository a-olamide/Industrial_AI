using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Risk
{
    public sealed record RiskSeriesResponse(
        string AssetId,
        DateTime From,
        DateTime To,
        IReadOnlyList<RiskPointDto> Points
    );
}

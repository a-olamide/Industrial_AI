using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Recommendations
{
    public sealed record Recommendation(
        string AssetId,
        DateTime AsOfMinuteTs,
        string RecType,
        byte Priority,
        string Title,
        string Description,
        decimal Confidence,
        string Drivers,
        string EvidenceJson,
        DateTime? CooldownUntil
    );
}

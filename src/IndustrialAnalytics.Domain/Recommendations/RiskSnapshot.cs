using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Recommendations
{
    public sealed record RiskSnapshot(
        string AssetId,
        DateTime AsOfMinuteTs,
        byte RiskScore,
        string RiskLevel,
        string? FailureMode,
        string TopDrivers
    );
}

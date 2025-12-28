using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Risk
{
    public sealed record RiskResult(
    string AssetId,
    DateTime MinuteTs,
    byte RiskScore,
    string RiskLevel,
    string? FailureMode,
    string TopDrivers,
    string EvidenceJson
);
}

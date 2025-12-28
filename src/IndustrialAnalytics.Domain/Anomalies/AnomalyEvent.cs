using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Anomalies
{
    public record AnomalyEvent(
    string AssetId,
    DateTime MinuteTs,
    string AnomalyType,
    string Signal,
    double? Score,
    byte Severity,
    string Reason,
    string? EvidenceJson
);
}

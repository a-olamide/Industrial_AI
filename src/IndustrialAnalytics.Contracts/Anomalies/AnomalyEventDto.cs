using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Anomalies
{
    public sealed record AnomalyEventDto(
        long AnomalyId,
        string AssetId,
        DateTime MinuteTs,
        string AnomalyType,
        string Signal,
        double Score,
        int Severity,
        string? Reason
    );
}

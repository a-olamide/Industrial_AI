using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Dtos
{
    public sealed record AnomalyDto(
        long AnomalyId,
        string AssetId,
        DateTime MinuteTs,
        string AnomalyType,
        string Signal,
        double Score,
        int Severity,
        string Reason,
        string? Evidence,      // keep as string so Dapper mapping is simple
        DateTime CreatedAt
    );
}

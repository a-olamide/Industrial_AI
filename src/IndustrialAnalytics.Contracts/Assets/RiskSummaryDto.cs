using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Assets
{
    public sealed record RiskSummaryDto(
        int Score,
        string Level,
        string? FailureMode,
        string? TopDrivers,
        object? Evidence
    );
}

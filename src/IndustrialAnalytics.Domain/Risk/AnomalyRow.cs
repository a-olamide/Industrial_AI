using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Risk
{
    public sealed record AnomalyRow(
    string AssetId,
    DateTime MinuteTs,
    string AnomalyType,
    string Signal,
    double? Score,
    byte Severity
);
}

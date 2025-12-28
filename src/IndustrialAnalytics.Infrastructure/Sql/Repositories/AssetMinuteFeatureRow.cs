using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{

    public sealed record AssetMinuteFeatureRow(
        string AssetId,
        DateTime MinuteTs,
        double? TempAvg15m,
        double? VibAvg15m,
        double? CurrentAvg15m,
        double? FlowAvg15m,
        double? PressureAvg15m,
        double? VibStd60m,
        double? TempSlope15m,
        double? VibSlope15m,
        double? CurrentSlope15m,
        double? FlowDropPct15m,
        int? RunMinutes60m,
        bool QualityGateOk
    );
}

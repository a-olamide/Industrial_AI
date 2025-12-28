using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Telemetry
{
    public sealed record FeatureSnapshot
    {
        public required string AssetId { get; init; }
        public required DateTime MinuteTs { get; init; }

        public bool QualityGateOk { get; init; }
        public int RunMinutes60m { get; init; }

        public double? TempAvg15m { get; init; }
        public double? VibAvg15m { get; init; }
        public double? CurrentAvg15m { get; init; }
        public double? FlowAvg15m { get; init; }
        public double? PressureAvg15m { get; init; }

        public double? VibStd60m { get; init; }
        public double? TempSlope15m { get; init; }
        public double? VibSlope15m { get; init; }
        public double? CurrentSlope15m { get; init; }

        public double? FlowDropPct15m { get; init; }
    }
}

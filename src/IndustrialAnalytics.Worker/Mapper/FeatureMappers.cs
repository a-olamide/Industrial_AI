using IndustrialAnalytics.Domain.Telemetry;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Worker.Mapper
{
    public static class FeatureMappers
    {
        public static FeatureSnapshot ToSnapshot(this AssetMinuteFeatureRow r) => new()
        {
            AssetId = r.AssetId,
            MinuteTs = r.MinuteTs,
            QualityGateOk = r.QualityGateOk,
            RunMinutes60m = r.RunMinutes60m ?? 0,

            TempAvg15m = r.TempAvg15m,
            VibAvg15m = r.VibAvg15m,
            CurrentAvg15m = r.CurrentAvg15m,
            FlowAvg15m = r.FlowAvg15m,
            PressureAvg15m = r.PressureAvg15m,

            VibStd60m = r.VibStd60m,
            TempSlope15m = r.TempSlope15m,
            VibSlope15m = r.VibSlope15m,
            CurrentSlope15m = r.CurrentSlope15m,

            FlowDropPct15m = r.FlowDropPct15m
        };
    }
}

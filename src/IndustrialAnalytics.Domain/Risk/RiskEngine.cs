using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Risk
{
    public sealed class RiskEngine : IRiskEngine
    {
        public RiskResult Compute(string assetId, DateTime minuteTs, IReadOnlyList<AnomalyRow> window, RiskOptions opt)
        {
            // Weighted sum with linear recency decay
            double score = 0;
            var driverCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            var contributions = new List<object>();

            foreach (var a in window)
            {
                var ageMin = (minuteTs - a.MinuteTs).TotalMinutes;
                if (ageMin < 0 || ageMin >= opt.WindowMinutes) continue;

                var decay = 1.0 - (ageMin / opt.WindowMinutes); // newest=1.0, oldest≈0
                var basePoints = PointsFor(a.AnomalyType, a.Severity, opt);
                var contrib = basePoints * decay;

                score += contrib;

                driverCounts[a.AnomalyType] = driverCounts.TryGetValue(a.AnomalyType, out var c) ? c + 1 : 1;
                contributions.Add(new { a.AnomalyType, a.Severity, a.MinuteTs, basePoints, decay, contrib });
            }

            var riskScore = (byte)Math.Clamp((int)Math.Round(score), 0, 100);

            var level = riskScore <= opt.LowMax ? "LOW"
                      : riskScore <= opt.MedMax ? "MED"
                      : "HIGH";

            var failureMode = ClassifyFailureMode(driverCounts);

            var topDrivers = string.Join(", ",
                driverCounts.OrderByDescending(kv => kv.Value).Take(3).Select(kv => $"{kv.Key}x{kv.Value}"));

            var evidence = JsonSerializer.Serialize(new
            {
                windowMinutes = opt.WindowMinutes,
                driverCounts,
                contributions = contributions.Take(50) // keep it bounded
            });

            return new RiskResult(assetId, minuteTs, riskScore, level, failureMode, topDrivers, evidence);
        }

        private static int PointsFor(string anomalyType, byte severity, RiskOptions opt) =>
            anomalyType switch
            {
                "TEMP_DRIFT" => severity >= 3 ? opt.TempDriftS3 : opt.TempDriftS2,
                "VIB_DRIFT" => severity >= 3 ? opt.VibDriftS3 : opt.VibDriftS2,
                "CURRENT_DRIFT" => severity >= 3 ? opt.CurrentDriftS3 : opt.CurrentDriftS2,
                "VIB_SPIKE" => severity >= 3 ? opt.VibSpikeS3 : opt.VibSpikeS2,
                "FLOW_DROP" => severity >= 3 ? opt.FlowDropS3 : opt.FlowDropS2,
                _ => 5
            };

        private static string? ClassifyFailureMode(Dictionary<string, int> counts)
        {
            bool temp = counts.ContainsKey("TEMP_DRIFT");
            bool vibD = counts.ContainsKey("VIB_DRIFT");
            bool vibS = counts.ContainsKey("VIB_SPIKE");
            bool flow = counts.ContainsKey("FLOW_DROP");
            bool curD = counts.ContainsKey("CURRENT_DRIFT") || counts.ContainsKey("CURRENT_HIGH");

            if (temp && vibD) return "BEARING_WEAR";
            if (flow && (vibS || vibD)) return "CAVITATION";
            if (flow && curD) return "OVERLOAD";
            return null;
        }
    }
}

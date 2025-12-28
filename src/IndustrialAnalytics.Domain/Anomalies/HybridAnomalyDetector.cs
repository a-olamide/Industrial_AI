using IndustrialAnalytics.Domain.Telemetry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Anomalies
{
    public sealed class HybridAnomalyDetector : IAnomalyDetector
    {
        public IReadOnlyList<AnomalyEvent> Detect(
            FeatureSnapshot cur,
            IReadOnlyList<FeatureSnapshot> lookback,
            Thresholds t)
        {
            var results = new List<AnomalyEvent>();

            // Gates
            if (!cur.QualityGateOk) return results;
            if (cur.RunMinutes60m < t.MinRunMinutes) return results;
            if (lookback.Count < t.MinRowsForBaseline) return results;

            // Use rolling averages as the signal value we score
            AddZScoreSpike(results, cur, lookback, t,
                signalName: Signals.Vibration,
                anomalyType: AnomalyTypes.VibSpike,
                valueSelector: x => x.VibAvg15m,
                zWarn: t.VibSpikeZ,
                zSevere: t.VibSevereZ,
                eps: t.VibStdEps);

            AddZScoreSpike(results, cur, lookback, t,
                signalName: Signals.Temperature,
                anomalyType: AnomalyTypes.TempHigh,
                valueSelector: x => x.TempAvg15m,
                zWarn: t.TempHighZ,
                zSevere: t.TempSevereZ,
                eps: t.TempStdEps);

            AddZScoreSpike(results, cur, lookback, t,
                signalName: Signals.Current,
                anomalyType: AnomalyTypes.CurrentHigh,
                valueSelector: x => x.CurrentAvg15m,
                zWarn: t.CurrentHighZ,
                zSevere: t.CurrentSevereZ,
                eps: t.CurrentStdEps);

            // Drift anomalies (slope persistence)
            AddSlopeDrift(results, cur, lookback, t,
                signalName: Signals.Vibration,
                anomalyType: AnomalyTypes.VibDrift,
                slopeSelector: x => x.VibSlope15m,
                driftSlope: t.VibDriftSlope,
                severeSlope: t.VibSevereSlope);

            AddSlopeDrift(results, cur, lookback, t,
                signalName: Signals.Temperature,
                anomalyType: AnomalyTypes.TempDrift,
                slopeSelector: x => x.TempSlope15m,
                driftSlope: t.TempDriftSlope,
                severeSlope: t.TempSevereSlope);

            AddSlopeDrift(results, cur, lookback, t,
                signalName: Signals.Current,
                anomalyType: AnomalyTypes.CurrentDrift,
                slopeSelector: x => x.CurrentSlope15m,
                driftSlope: t.CurrentDriftSlope,
                severeSlope: t.CurrentSevereSlope);

            // Flow drop rule (uses SQL feature when available)
            if (cur.FlowDropPct15m is double drop)
            {
                if (drop <= t.FlowDropPct)
                {
                    var severity = (byte)(drop <= t.FlowDropSeverePct ? 3 : 2);
                    results.Add(new AnomalyEvent(
                        cur.AssetId,
                        cur.MinuteTs,
                        AnomalyTypes.FlowDrop,
                        Signals.Flow,
                        drop,
                        severity,
                        $"Flow dropped {(drop * 100):0.#}% vs baseline.",
                        JsonSerializer.Serialize(new { flowDropPct = drop })
                    ));
                }
            }

            return results;
        }

        private static void AddZScoreSpike(
            List<AnomalyEvent> output,
            FeatureSnapshot cur,
            IReadOnlyList<FeatureSnapshot> lookback,
            Thresholds t,
            string signalName,
            string anomalyType,
            Func<FeatureSnapshot, double?> valueSelector,
            double zWarn,
            double zSevere,
            double eps)
        {
            var curVal = valueSelector(cur);
            if (curVal is null) return;

            var values = lookback
                .Select(valueSelector)
                .Where(v => v is not null)
                .Select(v => v!.Value)
                .ToList();

            if (values.Count < t.MinRowsForBaseline) return;

            var mean = values.Average();
            var std = StdDev(values);
            std = Math.Max(std, eps);

            var z = (curVal.Value - mean) / std;
            if (z < zWarn) return;

            var severity = (byte)(z >= zSevere ? 3 : 2);

            output.Add(new AnomalyEvent(
                cur.AssetId,
                cur.MinuteTs,
                anomalyType,
                signalName,
                z,
                severity,
                $"{signalName} spike: z={z:0.00} (val={curVal:0.###}, mean={mean:0.###}, std={std:0.###})",
                JsonSerializer.Serialize(new { val = curVal, mean, std, z })
            ));
        }

        private static void AddSlopeDrift(
    List<AnomalyEvent> output,
    FeatureSnapshot cur,
    IReadOnlyList<FeatureSnapshot> lookback,
    Thresholds t,
    string signalName,
    string anomalyType,
    Func<FeatureSnapshot, double?> slopeSelector,
    double driftSlope,
    double severeSlope)
        {
            var k = Math.Max(2, t.DriftConsecutiveMinutes);

            // Build a window of the last k minutes INCLUDING current minute,
            // but only if we have enough history.
            // Note: lookback already excludes current minute if you fixed the leakage.
            // So we take (k-1) from lookback + current.
            var prior = lookback
                .OrderByDescending(x => x.MinuteTs)
                .Take(k - 1)
                .ToList();

            if (prior.Count < k - 1) return;

            var window = new List<FeatureSnapshot>(k);
            window.Add(cur);
            window.AddRange(prior);

            // Ensure they are consecutive in time (minute granularity)
            // If you have gaps, don't call it "consecutive drift"
            var ordered = window.OrderBy(x => x.MinuteTs).ToList();
            for (int i = 1; i < ordered.Count; i++)
            {
                if ((ordered[i].MinuteTs - ordered[i - 1].MinuteTs).TotalMinutes != 1)
                    return;
            }

            var slopes = ordered
                .Select(slopeSelector)
                .ToList();

            // If any slope is null, we can't assert persistent drift
            if (slopes.Any(s => s is null)) return;

            var slopeVals = slopes.Select(s => s!.Value).ToList();

            // Check persistence above drift threshold
            if (slopeVals.All(s => s >= driftSlope))
            {
                // severity if all above severe threshold
                var severity = (byte)(slopeVals.All(s => s >= severeSlope) ? 3 : 2);

                var avgSlope = slopeVals.Average();
                output.Add(new AnomalyEvent(
                    cur.AssetId,
                    cur.MinuteTs,
                    anomalyType,
                    signalName,
                    avgSlope, // score = avg slope over window
                    severity,
                    $"{signalName} drift: slope >= {driftSlope:0.###} for {k} consecutive minutes (avgSlope={avgSlope:0.###}).",
                    System.Text.Json.JsonSerializer.Serialize(new
                    {
                        k,
                        driftSlope,
                        severeSlope,
                        slopes = slopeVals,
                        minutes = ordered.Select(x => x.MinuteTs)
                    })
                ));
            }
        }
        private static double StdDev(IReadOnlyList<double> xs)
        {
            if (xs.Count < 2) return 0.0;
            var mean = xs.Average();
            var sumSq = xs.Sum(x => (x - mean) * (x - mean));
            return Math.Sqrt(sumSq / (xs.Count - 1));
        }
    }
}

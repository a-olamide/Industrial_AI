using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Recommendations
{
    public sealed class RecommendationEngine : IRecommendationEngine
    {
        public IReadOnlyList<Recommendation> Generate(RiskSnapshot r, RecommendationOptions opt)
        {
            var recs = new List<Recommendation>();

            var cooldown = r.RiskLevel switch
            {
                "HIGH" => TimeSpan.FromHours(opt.CooldownHoursHigh),
                "MED" => TimeSpan.FromHours(opt.CooldownHoursMed),
                _ => TimeSpan.FromHours(opt.CooldownHoursLow)
            };

            var nowCooldownUntil = r.AsOfMinuteTs.Add(cooldown);

            var drivers = r.TopDrivers ?? "";
            var mode = r.FailureMode ?? "UNKNOWN";

            if (mode == "BEARING_WEAR")
            {
                if (r.RiskLevel == "HIGH")
                {
                    recs.Add(Make(
                        r, "BEARING_SCHEDULE_MAINT", 1,
                        "Schedule bearing inspection/replacement (72h)",
                        "High bearing wear risk detected. Schedule maintenance within 72 hours. Verify lubrication, alignment, and bearing condition. Consider reducing load until serviced.",
                        0.92m, drivers, nowCooldownUntil,
                        new { mode, r.RiskScore, r.RiskLevel, drivers }
                    ));
                }
                else if (r.RiskLevel == "MED")
                {
                    recs.Add(Make(
                        r, "BEARING_INSPECTION", 2,
                        "Inspect lubrication and alignment (next shift)",
                        "Moderate bearing wear trend detected. Inspect lubrication, coupling alignment, and bearing housing temperature on the next shift. Plan parts if trend persists.",
                        0.80m, drivers, nowCooldownUntil,
                        new { mode, r.RiskScore, r.RiskLevel, drivers }
                    ));
                }
            }
            else if (mode == "CAVITATION")
            {
                recs.Add(Make(
                    r, "CAVITATION_CHECK", r.RiskLevel == "HIGH" ? (byte)1 : (byte)2,
                    "Check suction conditions (cavitation suspected)",
                    "Signs consistent with cavitation. Check suction pressure/NPSH margin, ensure inlet valve is fully open, look for air ingress, and verify strainer condition.",
                    r.RiskLevel == "HIGH" ? 0.85m : 0.70m, drivers, nowCooldownUntil,
                    new { mode, r.RiskScore, r.RiskLevel, drivers }
                ));
            }
            else if (mode == "OVERLOAD")
            {
                recs.Add(Make(
                    r, "OVERLOAD_REDUCE_LOAD", r.RiskLevel == "HIGH" ? (byte)1 : (byte)2,
                    "Investigate overload / reduce load",
                    "Load trend suggests overload. Verify process demand, valve position, discharge pressure, and check for fouling. Consider reducing throughput until verified.",
                    r.RiskLevel == "HIGH" ? 0.82m : 0.68m, drivers, nowCooldownUntil,
                    new { mode, r.RiskScore, r.RiskLevel, drivers }
                ));
            }
            else
            {
                // Generic high risk fallback
                if (r.RiskLevel == "HIGH")
                {
                    recs.Add(Make(
                        r, "GENERAL_INSPECTION", 2,
                        "Perform general inspection (high risk)",
                        "High risk detected without a confident failure mode. Inspect bearing temp, vibration, coupling, lubrication, and review recent operating conditions.",
                        0.60m, drivers, nowCooldownUntil,
                        new { mode, r.RiskScore, r.RiskLevel, drivers }
                    ));
                }
            }

            return recs;
        }

        private static Recommendation Make(
            RiskSnapshot r,
            string recType,
            byte priority,
            string title,
            string description,
            decimal confidence,
            string drivers,
            DateTime cooldownUntil,
            object evidenceObj)
        {
            return new Recommendation(
                r.AssetId,
                r.AsOfMinuteTs,
                recType,
                priority,
                title,
                description,
                confidence,
                drivers,
                JsonSerializer.Serialize(evidenceObj),
                cooldownUntil
            );
        }
    }
}

using IndustrialAnalytics.Domain.Recommendations;
using IndustrialAnalytics.Infrastructure.Sql;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;
using IndustrialAnalytics.Worker.Options;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Worker.Workers
{
    public sealed class RecommendationWorker(
    IOptions<RecommendationWorkerOptions> wopt,
    IOptions<RecommendationOptions> ropt,
    ICheckpointStore checkpoint,
    IRiskQueryRepository riskQuery,
    IRecommendationEngine engine,
    IRecommendationRepository repo,
    ILogger<RecommendationWorker> log
) : BackgroundService
    {
        private readonly RecommendationWorkerOptions _w = wopt.Value;
        private readonly RecommendationOptions _opt = ropt.Value;

        // same safe epoch we used elsewhere
        private static readonly DateTime SafeEpochUtc = new(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            log.LogInformation("Starting {WorkerName}", _w.WorkerName);

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    var last = await checkpoint.GetLastAsync(_w.WorkerName, stoppingToken) ?? SafeEpochUtc;

                    // Pull current risk state (1 row per asset)
                    var current = await riskQuery.GetCurrentAsync(stoppingToken);

                    // Process only those updated since our checkpoint
                    var toProcess = current
                        //.Where(r => r.AsOfMinuteTs > last)
                        .OrderBy(r => r.AsOfMinuteTs)
                        .ToList();

                    if (toProcess.Count == 0)
                    {
                        await Task.Delay(TimeSpan.FromSeconds(_w.PollSeconds), stoppingToken);
                        continue;
                    }

                    DateTime maxProcessed = last;

                    foreach (var r in toProcess)
                    {
                        // AUTO-CLOSE POLICY: if risk is low or mode is gone, close active recs
                        var mode = r.FailureMode ?? "UNKNOWN";
                        if (r.RiskLevel == "LOW" || r.RiskScore < 30 || mode == "UNKNOWN")
                        {
                            // Always close generic recs if condition resolved
                            await repo.CloseActiveForAssetByPrefixAsync(
                                r.AssetId, "GENERAL_", "system",
                                $"Auto-closed (resolved): state={r.RiskLevel}/{r.RiskScore}, mode={mode}.",
                                stoppingToken);

                            // Also close the major families (safe, still “related”), because when risk is LOW, all should resolve.
                            await repo.CloseActiveForAssetByPrefixAsync(
                                r.AssetId, "BEARING_", "system",
                                $"Auto-closed (resolved): state={r.RiskLevel}/{r.RiskScore}, mode={mode}.",
                                stoppingToken);

                            await repo.CloseActiveForAssetByPrefixAsync(
                                r.AssetId, "CAVITATION_", "system",
                                $"Auto-closed (resolved): state={r.RiskLevel}/{r.RiskScore}, mode={mode}.",
                                stoppingToken);

                            await repo.CloseActiveForAssetByPrefixAsync(
                                r.AssetId, "OVERLOAD_", "system",
                                $"Auto-closed (resolved): state={r.RiskLevel}/{r.RiskScore}, mode={mode}.",
                                stoppingToken);

                            //await repo.CloseActiveForAssetAsync(
                            //    r.AssetId,
                            //    closedBy: "system",
                            //    reason: $"Auto-closed because state is {r.RiskLevel}/{r.RiskScore} mode={mode}.",
                            //    ct: stoppingToken);

                            // still advance checkpoint, but don't generate new recs for low/unknown state
                            continue;
                        }
                        var recs = engine.Generate(r, _opt);
                        var fp = Fingerprint(r);

                        foreach (var rec in recs)
                        {
                            var active = await repo.GetActiveStateAsync(rec.AssetId, rec.RecType, stoppingToken);

                            if (active is not null)
                            {
                                var activeFp = active.StateFingerprint ?? "";
                                var activeModePrefix =
                                    activeFp.StartsWith("BEARING_WEAR|", StringComparison.OrdinalIgnoreCase) ? "BEARING_" :
                                    activeFp.StartsWith("CAVITATION|", StringComparison.OrdinalIgnoreCase) ? "CAVITATION_" :
                                    activeFp.StartsWith("OVERLOAD|", StringComparison.OrdinalIgnoreCase) ? "OVERLOAD_" :
                                    "GENERAL_";

                                var currentPrefix = PrefixForMode(r.FailureMode);

                                // If the failure mode family changed, close the old family rec(s)
                                if (!string.Equals(activeModePrefix, currentPrefix, StringComparison.OrdinalIgnoreCase))
                                {
                                    await repo.CloseActiveForAssetByPrefixAsync(
                                        r.AssetId, activeModePrefix, "system",
                                        $"Auto-closed because failure mode changed to {r.FailureMode ?? "UNKNOWN"}.",
                                        stoppingToken);
                                }

                                // Existing ACK and fingerprint logic still applies
                                if (active.Status == "ACKED" && active.AckUntil is DateTime au && r.AsOfMinuteTs < au)
                                    continue;

                                if (string.Equals(active.StateFingerprint, fp, StringComparison.OrdinalIgnoreCase))
                                    continue;

                                await repo.CloseAsync(active.RecommendationId, "system", stoppingToken);
                            }

                            // Insert new recommendation (status OPEN), stored with fingerprint and cooldown_until
                            await repo.InsertAsync(rec, fp, stoppingToken);
                        }

                        if (r.AsOfMinuteTs > maxProcessed)
                            maxProcessed = r.AsOfMinuteTs;
                    }

                    // Advance checkpoint once per batch (faster + cleaner)
                    await checkpoint.SetLastAsync(_w.WorkerName, DateTime.UtcNow, stoppingToken);
                }
                catch (Exception ex)
                {
                    log.LogError(ex, "Recommendation worker loop error");
                    await Task.Delay(TimeSpan.FromSeconds(Math.Min(60, _w.PollSeconds * 2)), stoppingToken);
                }

                await Task.Delay(TimeSpan.FromSeconds(_w.PollSeconds), stoppingToken);
            }
        }

        private static string Fingerprint(RiskSnapshot r)
        {
            var mode = r.FailureMode ?? "UNKNOWN";
            var drivers = (r.TopDrivers ?? "").Trim();

            // bucket score into coarse band to avoid churn from tiny score changes
            var band = r.RiskScore >= 90 ? "90+"
                    : r.RiskScore >= 70 ? "70-89"
                    : r.RiskScore >= 30 ? "30-69"
                    : "0-29";

            return $"{mode}|{r.RiskLevel}|{band}|{drivers}";
        }

        private static string PrefixForMode(string? failureMode)
        {
            return (failureMode ?? "UNKNOWN") switch
            {
                "BEARING_WEAR" => "BEARING_",
                "CAVITATION" => "CAVITATION_",
                "OVERLOAD" => "OVERLOAD_",
                _ => "GENERAL_"
            };
        }
    }

}

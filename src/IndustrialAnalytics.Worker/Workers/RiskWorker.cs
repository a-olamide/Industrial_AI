using IndustrialAnalytics.Domain.Risk;
using IndustrialAnalytics.Infrastructure.Sql;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Worker.Workers
{
    public sealed record RiskWorkerOptions
    {
        public string WorkerName { get; init; } = "risk-worker";
        public int PollSeconds { get; init; } = 10;
    }

    public sealed class RiskWorker(
        IOptions<RiskWorkerOptions> wopt,
        IOptions<RiskOptions> ropt,
        ICheckpointStore checkpoint,
        IAnomalyQueryRepository anomalies,
        IRiskEngine engine,
        IRiskRepository riskRepo,
        ILogger<RiskWorker> log
    ) : BackgroundService
    {
        private readonly RiskWorkerOptions _w = wopt.Value;
        private readonly RiskOptions _r = ropt.Value;

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            log.LogInformation("Starting {WorkerName}", _w.WorkerName);

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    var last = await checkpoint.GetLastAsync(_w.WorkerName, stoppingToken)
                               ?? new DateTime(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc);

                    // Find which (asset, minute) have anomalies since last checkpoint
                    var minutes = await anomalies.GetDistinctMinutesSinceAsync(last, stoppingToken);

                    if (minutes.Count == 0)
                    {
                        await Task.Delay(TimeSpan.FromSeconds(_w.PollSeconds), stoppingToken);
                        continue;
                    }

                    foreach (var (assetId, minuteTs) in minutes)
                    {
                        // Window ends at minuteTs + 1 minute (exclusive) so it includes that minute’s anomalies
                        var toExclusive = minuteTs.AddMinutes(1);

                        var window = await anomalies.GetWindowAsync(assetId, toExclusive, _r.WindowMinutes, stoppingToken);

                        var result = engine.Compute(assetId, minuteTs, window, _r);

                        await riskRepo.UpsertMinuteAsync(result, stoppingToken);
                        await riskRepo.UpsertCurrentAsync(result, stoppingToken);

                        await checkpoint.SetLastAsync(_w.WorkerName, minuteTs, stoppingToken);
                    }
                }
                catch (Exception ex)
                {
                    log.LogError(ex, "Risk worker loop error");
                    await Task.Delay(TimeSpan.FromSeconds(Math.Min(30, _w.PollSeconds * 3)), stoppingToken);
                }
            }
        }
    }
}

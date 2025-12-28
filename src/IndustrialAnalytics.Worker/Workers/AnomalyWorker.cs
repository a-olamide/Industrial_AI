using IndustrialAnalytics.Domain.Anomalies;
using IndustrialAnalytics.Infrastructure.Sql;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;
using IndustrialAnalytics.Worker.Mapper;
using IndustrialAnalytics.Worker.Options;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Worker.Workers
{
    public sealed class AnomalyWorker(
    IOptions<WorkerOptions> workerOpt,
    IOptions<Thresholds> thresholdsOpt,
    ICheckpointStore checkpoint,
    IFeatureRepository featureRepo,
    IAnomalyDetector detector,
    IAnomalyEventRepository anomalyRepo,
    ILogger<AnomalyWorker> log
) : BackgroundService
    {
        private readonly WorkerOptions _worker = workerOpt.Value;
        private readonly Thresholds _t = thresholdsOpt.Value;

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            log.LogInformation("Starting {WorkerName}", _worker.Name);

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    var last = await checkpoint.GetLastAsync(_worker.Name, stoppingToken) ?? new DateTime(2000, 1, 1, 0, 0, 0, DateTimeKind.Utc);

                    var newRows = await featureRepo.GetNewFeaturesAsync(last, stoppingToken);
                    if (newRows.Count == 0)
                    {
                        await Task.Delay(TimeSpan.FromSeconds(_worker.PollSeconds), stoppingToken);
                        continue;
                    }

                    foreach (var curRow in newRows.OrderBy(r => r.MinuteTs))
                    {
                        var lookbackRows = await featureRepo.GetLookbackAsync(
                            curRow.AssetId,
                            curRow.MinuteTs,
                            _t.LookbackMinutesDev,
                            stoppingToken);

                        var cur = curRow.ToSnapshot();
                        var lookback = lookbackRows.Select(r => r.ToSnapshot()).ToList();

                        var anomalies = detector.Detect(cur, lookback, _t);

                        if (anomalies.Count > 0)
                            await anomalyRepo.InsertManyAsync(anomalies, stoppingToken);

                        await checkpoint.SetLastAsync(_worker.Name, curRow.MinuteTs, stoppingToken);
                    }
                }
                catch (Exception ex)
                {
                    log.LogError(ex, "Worker loop error");
                    await Task.Delay(TimeSpan.FromSeconds(Math.Min(30, _worker.PollSeconds * 3)), stoppingToken);
                }
            }
        }
    }
}

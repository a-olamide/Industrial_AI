using Dapper;
using IndustrialAnalytics.Domain.Telemetry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public sealed class FeatureRepository(ISqlConnectionFactory f) : IFeatureRepository
    {
        public async Task<IReadOnlyList<AssetMinuteFeatureRow>> GetNewFeaturesAsync(DateTime fromExclusive, CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                                SELECT
                                  asset_id AS AssetId,
                                  minute_ts AS MinuteTs,
                                  temp_avg_15m AS TempAvg15m,
                                  vib_avg_15m AS VibAvg15m,
                                  current_avg_15m AS CurrentAvg15m,
                                  flow_avg_15m AS FlowAvg15m,
                                  pressure_avg_15m AS PressureAvg15m,
                                  vib_std_60m AS VibStd60m,
                                  temp_slope_15m AS TempSlope15m,
                                  vib_slope_15m AS VibSlope15m,
                                  current_slope_15m AS CurrentSlope15m,
                                  flow_drop_pct_15m AS FlowDropPct15m,
                                  run_minutes_60m AS RunMinutes60m,
                                  quality_gate_ok AS QualityGateOk
                                FROM dbo.asset_minute_features
                                WHERE minute_ts > @fromExclusive
                                ORDER BY minute_ts, asset_id;";

            var rows = await conn.QueryAsync<AssetMinuteFeatureRow>(
                new CommandDefinition(sql, new { fromExclusive }, cancellationToken: ct));
            return rows.AsList();
        }

        public async Task<IReadOnlyList<AssetMinuteFeatureRow>> GetLookbackAsync(string assetId, DateTime upToInclusive, int minutes, CancellationToken ct)
        {
            using var conn = f.Create();
            var from = upToInclusive.AddMinutes(-minutes);

            const string sql = @"
                                SELECT TOP (@maxRows)
                                  asset_id AS AssetId,
                                  minute_ts AS MinuteTs,
                                  temp_avg_15m AS TempAvg15m,
                                  vib_avg_15m AS VibAvg15m,
                                  current_avg_15m AS CurrentAvg15m,
                                  flow_avg_15m AS FlowAvg15m,
                                  pressure_avg_15m AS PressureAvg15m,
                                  vib_std_60m AS VibStd60m,
                                  temp_slope_15m AS TempSlope15m,
                                  vib_slope_15m AS VibSlope15m,
                                  current_slope_15m AS CurrentSlope15m,
                                  flow_drop_pct_15m AS FlowDropPct15m,
                                  run_minutes_60m AS RunMinutes60m,
                                  quality_gate_ok AS QualityGateOk
                                FROM dbo.asset_minute_features
                                WHERE asset_id = @assetId
                                  AND minute_ts >= @from
                                  AND minute_ts < @upToInclusive
                                  AND quality_gate_ok = 1
                                ORDER BY minute_ts DESC;";

            var rows = await conn.QueryAsync<AssetMinuteFeatureRow>(
                new CommandDefinition(sql, new { assetId, from, upToInclusive, maxRows = minutes + 5 }, cancellationToken: ct));
            return rows.AsList();
        }
    }
}

using Dapper;
using IndustrialAnalytics.Contracts.Dtos;
using IndustrialAnalytics.Domain.Risk;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public sealed class AnomalyQueryRepository(ISqlConnectionFactory f) : IAnomalyQueryRepository
    {
        public async Task<IReadOnlyList<(string AssetId, DateTime MinuteTs)>> GetDistinctMinutesSinceAsync(
            DateTime fromExclusive,
            CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                                SELECT DISTINCT asset_id AS AssetId, minute_ts AS MinuteTs
                                FROM dbo.asset_anomaly_events
                                WHERE minute_ts > @fromExclusive
                                ORDER BY minute_ts, asset_id;";

            var rows = await conn.QueryAsync<(string AssetId, DateTime MinuteTs)>(
                new CommandDefinition(sql, new { fromExclusive }, cancellationToken: ct));
            return rows.AsList();
        }

        public async Task<IReadOnlyList<AnomalyRow>> GetWindowAsync(
            string assetId,
            DateTime toExclusive,
            int windowMinutes,
            CancellationToken ct)
        {
            using var conn = f.Create();
            var from = toExclusive.AddMinutes(-windowMinutes);

            const string sql = @"
                            SELECT asset_id AS AssetId, minute_ts AS MinuteTs, anomaly_type AS AnomalyType,
                                   signal AS Signal, score AS Score, severity AS Severity
                            FROM dbo.asset_anomaly_events
                            WHERE asset_id = @assetId
                              AND minute_ts >= @from
                              AND minute_ts <  @toExclusive
                            ORDER BY minute_ts;";

            var rows = await conn.QueryAsync<AnomalyRow>(
                new CommandDefinition(sql, new { assetId, from, toExclusive }, cancellationToken: ct));
            return rows.AsList();
        }

        public async Task<IReadOnlyList<AnomalyDto>> GetForAssetAsync(
            string assetId,
            DateTime fromUtc,
            DateTime toUtc,
            int take,
            CancellationToken ct)
        {
            const string sql = @"
            SELECT TOP (@take)
                anomaly_id AS AnomalyId,
                asset_id   AS AssetId,
                minute_ts  AS MinuteTs,
                anomaly_type AS AnomalyType,
                signal     AS Signal,
                CAST(score AS float) AS Score,
                CAST(severity AS int) AS Severity,
                reason     AS Reason,
                CAST(evidence_json AS nvarchar(max)) AS Evidence,
                created_at AS CreatedAt
            FROM dbo.asset_anomaly_events
            WHERE asset_id = @assetId
              AND minute_ts >= @fromUtc
              AND minute_ts <  @toUtc
            ORDER BY minute_ts;";

            using var conn = f.Create();
            var rows = await conn.QueryAsync<AnomalyRow>(
                new CommandDefinition(sql, new { assetId, fromUtc, toUtc, take }, cancellationToken: ct));

            return rows.Select(r => new AnomalyDto(
                r.AnomalyId,
                r.AssetId,
                r.MinuteTs,
                r.AnomalyType,
                r.Signal,
                r.Score,
                r.Severity,
                r.Reason,
                r.Evidence,
                r.CreatedAt
            )).ToList();
        }

      

    }

}

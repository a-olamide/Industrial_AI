using Dapper;
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
    }
}

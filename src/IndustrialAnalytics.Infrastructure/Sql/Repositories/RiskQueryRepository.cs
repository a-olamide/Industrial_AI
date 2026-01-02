using Dapper;
using IndustrialAnalytics.Contracts.Risk;
using IndustrialAnalytics.Domain.Recommendations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public sealed class RiskQueryRepository(ISqlConnectionFactory f) : IRiskQueryRepository
    {
        public async Task<IReadOnlyList<RiskSnapshot>> GetCurrentAsync(CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                    SELECT
                      asset_id AS AssetId,
                      as_of_minute_ts AS AsOfMinuteTs,
                      risk_score AS RiskScore,
                      risk_level AS RiskLevel,
                      failure_mode AS FailureMode,
                      top_drivers AS TopDrivers
                    FROM dbo.asset_risk_current;";

            var rows = await conn.QueryAsync<RiskSnapshot>(
                new CommandDefinition(sql, cancellationToken: ct));
            return rows.AsList();
        }

        public async Task<RiskSeriesResponse> GetRiskSeriesAsync(
        string assetId, DateTimeOffset from, DateTimeOffset to, int? stepMinutes, CancellationToken ct)
        {
            // stepMinutes 1/5/15 supported; if >1, sample using grouping
            stepMinutes = stepMinutes <= 1 ? 1 : stepMinutes;

            string sql = stepMinutes == 1
                ? @"
                SELECT
                  minute_ts AS MinuteTs,
                  risk_score AS Score,
                  risk_level AS Level,
                  failure_mode AS FailureMode
                FROM dbo.asset_risk_minute
                WHERE asset_id = @assetId
                  AND minute_ts >= @from AND minute_ts < @to
                ORDER BY minute_ts;"
                                : @"
                WITH b AS (
                  SELECT
                    DATEADD(minute, (DATEDIFF(minute, 0, minute_ts) / @stepMinutes) * @stepMinutes, 0) AS bucket_ts,
                    risk_score, risk_level, failure_mode
                  FROM dbo.asset_risk_minute
                  WHERE asset_id = @assetId
                    AND minute_ts >= @from AND minute_ts < @to
                )
                SELECT
                  bucket_ts AS MinuteTs,
                  MAX(risk_score) AS Score,
                  MAX(risk_level) AS Level,
                  MAX(failure_mode) AS FailureMode
                FROM b
                GROUP BY bucket_ts
                ORDER BY bucket_ts;";

            using var conn = f.Create();
            var points = (await conn.QueryAsync<RiskPointDto>(new CommandDefinition(
                sql, new { assetId, from, to, stepMinutes }, cancellationToken: ct))).AsList();

            return new RiskSeriesResponse(assetId, from.UtcDateTime, to.UtcDateTime, points);
        }
    }
}

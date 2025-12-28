using Dapper;
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
    }
}

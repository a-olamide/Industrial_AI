using Dapper;
using IndustrialAnalytics.Domain.Risk;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public sealed class RiskRepository(ISqlConnectionFactory f) : IRiskRepository
    {
        public async Task UpsertMinuteAsync(RiskResult r, CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                            MERGE dbo.asset_risk_minute AS tgt
                            USING (SELECT @AssetId AS asset_id, @MinuteTs AS minute_ts) AS src
                            ON tgt.asset_id = src.asset_id AND tgt.minute_ts = src.minute_ts
                            WHEN MATCHED THEN UPDATE SET
                              risk_score=@RiskScore, risk_level=@RiskLevel, failure_mode=@FailureMode,
                              top_drivers=@TopDrivers, evidence_json=@EvidenceJson, computed_at=SYSUTCDATETIME()
                            WHEN NOT MATCHED THEN INSERT
                              (asset_id, minute_ts, risk_score, risk_level, failure_mode, top_drivers, evidence_json)
                            VALUES
                              (@AssetId, @MinuteTs, @RiskScore, @RiskLevel, @FailureMode, @TopDrivers, @EvidenceJson);";

            await conn.ExecuteAsync(new CommandDefinition(sql, new
            {
                r.AssetId,
                r.MinuteTs,
                r.RiskScore,
                r.RiskLevel,
                r.FailureMode,
                r.TopDrivers,
                r.EvidenceJson
            }, cancellationToken: ct));
        }

        public async Task UpsertCurrentAsync(RiskResult r, CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                        MERGE dbo.asset_risk_current AS tgt
                        USING (SELECT @AssetId AS asset_id) AS src
                        ON tgt.asset_id = src.asset_id
                        WHEN MATCHED THEN UPDATE SET
                          as_of_minute_ts=@MinuteTs, risk_score=@RiskScore, risk_level=@RiskLevel, failure_mode=@FailureMode,
                          top_drivers=@TopDrivers, evidence_json=@EvidenceJson, updated_at=SYSUTCDATETIME()
                        WHEN NOT MATCHED THEN INSERT
                          (asset_id, as_of_minute_ts, risk_score, risk_level, failure_mode, top_drivers, evidence_json)
                        VALUES
                          (@AssetId, @MinuteTs, @RiskScore, @RiskLevel, @FailureMode, @TopDrivers, @EvidenceJson);";

            await conn.ExecuteAsync(new CommandDefinition(sql, new
            {
                r.AssetId,
                r.MinuteTs,
                r.RiskScore,
                r.RiskLevel,
                r.FailureMode,
                r.TopDrivers,
                r.EvidenceJson
            }, cancellationToken: ct));
        }
    }
}

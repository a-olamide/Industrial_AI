using Dapper;
using IndustrialAnalytics.Domain.Recommendations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public sealed class RecommendationRepository(ISqlConnectionFactory f) : IRecommendationRepository
    {
        public async Task<RecommendationState?> GetActiveStateAsync(string assetId, string recType, CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                SELECT TOP 1
                  recommendation_id AS RecommendationId,
                  status AS Status,
                  ack_until AS AckUntil,
                  state_fingerprint AS StateFingerprint
                FROM dbo.asset_recommendations
                WHERE asset_id=@assetId
                  AND rec_type=@recType
                  AND status <> 'CLOSED'
                ORDER BY created_at DESC;";

            return await conn.QuerySingleOrDefaultAsync<RecommendationState>(
                new CommandDefinition(sql, new { assetId, recType }, cancellationToken: ct));
        }

        public async Task InsertAsync(Recommendation r, string stateFingerprint, CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                INSERT INTO dbo.asset_recommendations
                  (asset_id, as_of_minute_ts, rec_type, priority, title, description, status, confidence, drivers, evidence_json, cooldown_until, state_fingerprint)
                VALUES
                  (@AssetId, @AsOfMinuteTs, @RecType, @Priority, @Title, @Description, 'OPEN', @Confidence, @Drivers, @EvidenceJson, @CooldownUntil, @StateFingerprint);";

            await conn.ExecuteAsync(new CommandDefinition(sql, new
            {
                r.AssetId,
                r.AsOfMinuteTs,
                r.RecType,
                r.Priority,
                r.Title,
                r.Description,
                r.Confidence,
                r.Drivers,
                r.EvidenceJson,
                r.CooldownUntil,
                StateFingerprint = stateFingerprint
            }, cancellationToken: ct));
        }

        public async Task AcknowledgeAsync(long recommendationId, string acknowledgedBy, DateTime ackUntil, CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                UPDATE dbo.asset_recommendations
                SET status='ACKED',
                    acknowledged_at = SYSUTCDATETIME(),
                    acknowledged_by = @ackBy,
                    ack_until = @ackUntil,
                    updated_at = SYSUTCDATETIME()
                WHERE recommendation_id=@id
                  AND status <> 'CLOSED';";

            await conn.ExecuteAsync(new CommandDefinition(sql, new { id = recommendationId, ackBy = acknowledgedBy, ackUntil }, cancellationToken: ct));
        }

        public async Task CloseAsync(long recommendationId, string closedBy, CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                UPDATE dbo.asset_recommendations
                SET status='CLOSED',
                    closed_at = SYSUTCDATETIME(),
                    closed_by = @closedBy,
                    updated_at = SYSUTCDATETIME()
                WHERE recommendation_id=@id
                  AND status <> 'CLOSED';";

            await conn.ExecuteAsync(new CommandDefinition(sql, new { id = recommendationId, closedBy }, cancellationToken: ct));
        }
        public async Task CloseActiveForAssetAsync(string assetId, string closedBy, string reason, CancellationToken ct)
        {
            using var conn = f.Create();
            const string sql = @"
                UPDATE dbo.asset_recommendations
                SET status='CLOSED',
                    closed_at = SYSUTCDATETIME(),
                    closed_by = @closedBy,
                    updated_at = SYSUTCDATETIME(),
                    evidence_json = CASE
                        WHEN evidence_json IS NULL OR ISJSON(evidence_json) = 0
                        THEN JSON_OBJECT('autoCloseReason', @reason)
                        ELSE JSON_MODIFY(evidence_json, '$.autoCloseReason', @reason)
                    END
                WHERE asset_id=@assetId
                  AND status <> 'CLOSED';";

            await conn.ExecuteAsync(new CommandDefinition(sql, new { assetId, closedBy, reason }, cancellationToken: ct));
        }

        public async Task CloseActiveForAssetByPrefixAsync(
            string assetId,
            string recTypePrefix,
            string closedBy,
            string reason,
            CancellationToken ct)
            {
                using var conn = f.Create();

                const string sql = @"
                    UPDATE dbo.asset_recommendations
                    SET
                        status = 'CLOSED',
                        closed_at = SYSUTCDATETIME(),
                        closed_by = @closedBy,
                        updated_at = SYSUTCDATETIME(),
                        evidence_json =
                            JSON_MODIFY(
                                CASE
                                    WHEN evidence_json IS NULL OR ISJSON(evidence_json) = 0
                                        THEN '{}'
                                    ELSE evidence_json
                                END,
                                '$.autoCloseReason',
                                @reason
                            )
                    WHERE asset_id = @assetId
                      AND status <> 'CLOSED'
                      AND rec_type LIKE @recTypeLike;
                    ";

                await conn.ExecuteAsync(
                    new CommandDefinition(
                        sql,
                        new
                        {
                            assetId,
                            recTypeLike = recTypePrefix + "%",
                            closedBy,
                            reason
                        },
                        cancellationToken: ct
                    )
                );
            }

    }
}

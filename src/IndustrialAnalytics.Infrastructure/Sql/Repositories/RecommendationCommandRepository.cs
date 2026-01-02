using Dapper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public sealed class RecommendationCommandRepository(ISqlConnectionFactory f) : IRecommendationCommandRepository
    {
        public async Task<bool> AckAsync(long id, DateTime? ackUntil, string by, string? note, CancellationToken ct)
        {
            // Status becomes ACKED; store ack_until; append evidence_json note (best effort)
          

            const string sql = @"
            UPDATE dbo.asset_recommendations
            SET
                status = 'ACKED',
                ack_until = COALESCE(@ackUntil, DATEADD(day, 1, SYSUTCDATETIME())),
                updated_at = SYSUTCDATETIME(),
                evidence_json =
                    CASE
                        WHEN evidence_json IS NULL OR ISJSON(evidence_json) = 0
                            THEN JSON_MODIFY(
                                    JSON_MODIFY('{}', '$.ackNote', @note),
                                    '$.ackedBy', @by
                                 )
                        ELSE JSON_MODIFY(
                                JSON_MODIFY(evidence_json, '$.ackNote', @note),
                                '$.ackedBy', @by
                             )
                    END
            WHERE recommendation_id = @id
              AND status IN ('OPEN', 'ACKED');
            ";

            using var conn = f.Create();
            var affected = await conn.ExecuteAsync(new CommandDefinition(sql, new { id, ackUntil, by, note }, cancellationToken: ct));
            return affected > 0;
        }

        public async Task<bool> CloseAsync(long id, string reason, string by, CancellationToken ct)
        {
            const string sql = @"
                UPDATE dbo.asset_recommendations
                SET
                    status = 'CLOSED',
                    closed_at = SYSUTCDATETIME(),
                    closed_by = @by,
                    updated_at = SYSUTCDATETIME(),
                    evidence_json =
                        CASE
                            WHEN evidence_json IS NULL OR ISJSON(evidence_json) = 0
                                THEN JSON_MODIFY(
                                        JSON_MODIFY('{}', '$.closeReason', @reason),
                                        '$.closedBy', @by
                                     )
                            ELSE JSON_MODIFY(
                                    JSON_MODIFY(evidence_json, '$.closeReason', @reason),
                                    '$.closedBy', @by
                                 )
                        END
                WHERE recommendation_id = @id
                  AND status<> 'CLOSED';
                            ";


        using var conn = f.Create();
            var affected = await conn.ExecuteAsync(new CommandDefinition(sql, new { id, reason, by }, cancellationToken: ct));
            return affected > 0;
        }
    }
}

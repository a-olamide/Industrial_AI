using Dapper;
using IndustrialAnalytics.Contracts.Recommendations;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public sealed class RecommendationQueryRepository(ISqlConnectionFactory f)
        : IRecommendationQueryRepository
    {
        public async Task<IReadOnlyList<RecommendationDto>> GetQueueAsync(
        string status, int take, string? assetId, CancellationToken ct)
        {
            const string sql = @"
                SELECT TOP (@take)
                    recommendation_id      AS RecommendationId,
                    asset_id               AS AssetId,
                    as_of_minute_ts        AS AsOfMinuteTs,
                    rec_type               AS RecType,
                    CAST(priority AS int)               AS Priority,
                    title                  AS Title,
                    description            AS Description,
                    status                 AS Status,
                    CAST(confidence AS float)              AS Confidence,
                    CAST(drivers AS nvarchar(max))                AS Drivers,
                    evidence_json          AS Evidence,
                    created_at             AS CreatedAt,
                    updated_at             AS UpdatedAt,
                    CAST(ack_until AS datetime2)              AS AckUntil
                FROM dbo.asset_recommendations
                WHERE status = @status
                  AND (@assetId IS NULL OR asset_id = @assetId)
                ORDER BY as_of_minute_ts DESC, created_at DESC;
                ";

            using var conn = f.Create();
            var rows = await conn.QueryAsync<RecommendationDto>(
                new CommandDefinition(sql, new { status, take, assetId }, cancellationToken: ct));

            return rows.AsList();
        }

        public async Task<IReadOnlyList<RecommendationDto>> GetForAssetAsync(
            string assetId,
            string? status,
            int take,
            CancellationToken ct)
        {
            if (string.IsNullOrWhiteSpace(assetId))
                throw new ArgumentException("assetId is required", nameof(assetId));

            take = Math.Clamp(take <= 0 ? 50 : take, 1, 500);

            const string sql = @"
            SELECT TOP (@take)
                recommendation_id AS RecommendationId,
                asset_id          AS AssetId,
                as_of_minute_ts   AS AsOfMinuteTs,
                rec_type          AS RecType,
                CAST(priority AS int)          AS Priority,
                title             AS Title,
                description       AS Description,
                status            AS Status,
                CAST(confidence AS float)        AS Confidence,
                CAST(drivers AS nvarchar(max))           AS Drivers,
                CAST(evidence_json AS nvarchar(max))     AS Evidence,
                created_at        AS CreatedAt,
                updated_at        AS UpdatedAt,
                CAST(ack_until AS datetime2)         AS AckUntil
            FROM dbo.asset_recommendations
            WHERE asset_id = @assetId
              AND (@status IS NULL OR status = @status)
            ORDER BY created_at DESC;";

            using var conn = f.Create();
            if (conn.State != ConnectionState.Open)
                await ((dynamic)conn).OpenAsync(ct);

            var rows = await conn.QueryAsync<RecommendationDto>(
                new CommandDefinition(sql, new { assetId, status, take }, cancellationToken: ct));

            return rows.AsList();
        }
    }
}

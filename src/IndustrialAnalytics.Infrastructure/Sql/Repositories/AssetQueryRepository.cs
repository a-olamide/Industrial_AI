using Dapper;
using IndustrialAnalytics.Contracts.Assets;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public sealed class AssetQueryRepository(ISqlConnectionFactory f) : IAssetQueryRepository
    {
        public async Task<AssetListResponse> GetAssetsAsync(string? q, int limit, string? cursor, CancellationToken ct)
        {
            // cursor = last asset_id returned (simple, stable)
            var after = cursor ?? "";

            const string sql = @"
                SELECT TOP (@limit)
                    arc.asset_id AS AssetId,
                    CAST(NULL AS nvarchar(32)) AS AssetType,
                    CAST(NULL AS nvarchar(64)) AS Site,
                    arc.asset_id AS DisplayName,
                    arc.as_of_minute_ts AS LastSeenMinuteTs
                FROM dbo.asset_risk_current arc
                WHERE arc.asset_id > @after
                  AND (@q IS NULL OR arc.asset_id LIKE '%' + @q + '%')
                ORDER BY arc.asset_id;";

            using var conn = f.Create();
            var rows = (await conn.QueryAsync<AssetDto>(new CommandDefinition(
                sql, new { q, limit, after }, cancellationToken: ct))).AsList();

            var nextCursor = rows.Count == limit ? rows[^1].AssetId : null;
            return new AssetListResponse(rows, nextCursor);
        }

        public async Task<AssetSummaryDto?> GetAssetSummaryAsync(string assetId, CancellationToken ct)
        {
            const string sql = @"
                SELECT
                  arc.asset_id AS AssetId,
                  arc.as_of_minute_ts AS AsOfMinuteTs,
                  arc.risk_score AS RiskScore,
                  arc.risk_level AS RiskLevel,
                  arc.failure_mode AS FailureMode,
                  arc.top_drivers AS TopDrivers,
                  arc.evidence_json AS EvidenceJson,
                  (SELECT COUNT(1) FROM dbo.asset_recommendations r WHERE r.asset_id = arc.asset_id AND r.status = 'OPEN')  AS OpenCount,
                  (SELECT COUNT(1) FROM dbo.asset_recommendations r WHERE r.asset_id = arc.asset_id AND r.status = 'ACKED') AS AckedCount
                FROM dbo.asset_risk_current arc
                WHERE arc.asset_id = @assetId;";

            using var conn = f.Create();
            var row = await conn.QuerySingleOrDefaultAsync(sql, new { assetId });
            if (row is null) return null;

            object? evidence = null;
            try
            {
                if (row.EvidenceJson is string s && !string.IsNullOrWhiteSpace(s))
                    evidence = JsonSerializer.Deserialize<object>(s);
            }
            catch { /* best-effort */ }

            return new AssetSummaryDto(
                assetId,
                (DateTime)row.AsOfMinuteTs,
                new RiskSummaryDto((int)row.RiskScore, (string)row.RiskLevel, (string?)row.FailureMode, (string?)row.TopDrivers, evidence),
                new RecommendationCountsDto((int)row.OpenCount, (int)row.AckedCount)
            );
        }
    }
}

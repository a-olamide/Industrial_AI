using IndustrialAnalytics.Contracts.Assets;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public interface IAssetQueryRepository
    {
        Task<AssetListResponse> GetAssetsAsync(string? q, int limit, string? cursor, CancellationToken ct);
        Task<AssetSummaryDto?> GetAssetSummaryAsync(string assetId, CancellationToken ct);
    }
}

using IndustrialAnalytics.Contracts.Recommendations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public interface IRecommendationQueryRepository
    {
        Task<IReadOnlyList<RecommendationDto>> GetQueueAsync(
            string status,
            int take,
            string? assetId,
            CancellationToken ct);
        Task<IReadOnlyList<RecommendationDto>> GetForAssetAsync(
            string assetId,
            string? status,
            int take,
            CancellationToken ct);
    }
}

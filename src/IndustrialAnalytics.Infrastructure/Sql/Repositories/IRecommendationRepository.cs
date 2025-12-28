using IndustrialAnalytics.Domain.Recommendations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{

    public interface IRecommendationRepository
    {
        Task<RecommendationState?> GetActiveStateAsync(string assetId, string recType, CancellationToken ct);
        Task InsertAsync(Recommendation rec, string stateFingerprint, CancellationToken ct);

        Task AcknowledgeAsync(long recommendationId, string acknowledgedBy, DateTime ackUntil, CancellationToken ct);
        Task CloseAsync(long recommendationId, string closedBy, CancellationToken ct);
        Task CloseActiveForAssetAsync(string assetId, string closedBy, string reason, CancellationToken ct);
        Task CloseActiveForAssetByPrefixAsync(string assetId, string recTypePrefix, string closedBy, string reason, CancellationToken ct);

    }

    public sealed record RecommendationState(
        long RecommendationId,
        string Status,                 // OPEN / ACKED
        DateTime? AckUntil,
        string? StateFingerprint
    );
}

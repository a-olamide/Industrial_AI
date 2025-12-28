using IndustrialAnalytics.Domain.Risk;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public interface IAnomalyQueryRepository
    {
        Task<IReadOnlyList<AnomalyRow>> GetWindowAsync(
            string assetId,
            DateTime toExclusive,
            int windowMinutes,
            CancellationToken ct);

        Task<IReadOnlyList<(string AssetId, DateTime MinuteTs)>> GetDistinctMinutesSinceAsync(
            DateTime fromExclusive,
            CancellationToken ct);
    }
}

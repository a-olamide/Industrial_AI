using IndustrialAnalytics.Contracts.Risk;
using IndustrialAnalytics.Domain.Recommendations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public interface IRiskQueryRepository
    {
        Task<IReadOnlyList<RiskSnapshot>> GetCurrentAsync(CancellationToken ct);
        Task<RiskSeriesResponse> GetRiskSeriesAsync(string assetId, DateTimeOffset from, DateTimeOffset to, int? stepMinutes, CancellationToken ct);

    }
}

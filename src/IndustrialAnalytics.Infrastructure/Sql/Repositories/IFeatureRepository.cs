using IndustrialAnalytics.Domain.Telemetry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public interface IFeatureRepository
    {
        Task<IReadOnlyList<AssetMinuteFeatureRow>> GetNewFeaturesAsync(DateTime fromExclusive, CancellationToken ct);
        Task<IReadOnlyList<AssetMinuteFeatureRow>> GetLookbackAsync(string assetId, DateTime upToInclusive, int minutes, CancellationToken ct);
    }
}

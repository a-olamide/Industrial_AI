using IndustrialAnalytics.Contracts.DigitalTwins;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    /// <summary>
    /// Read access to the ML-derived Digital Twin state written by the
    /// Spark vibration inference job. Read-only by design: the twin is
    /// owned by the streaming pipeline, and nothing in the .NET stack
    /// writes to it.
    /// </summary>
    public interface IDigitalTwinQueryRepository
    {
        Task<IReadOnlyList<DigitalTwinStateDto>> GetAllAsync(
            bool includeGroundTruth,
            CancellationToken ct);

        Task<DigitalTwinStateDto?> GetByAssetAsync(
            string assetId,
            bool includeGroundTruth,
            CancellationToken ct);

        Task<IReadOnlyList<DigitalTwinHistoryPointDto>> GetHistoryAsync(
            string assetId,
            int take,
            bool includeGroundTruth,
            CancellationToken ct);
    }
}

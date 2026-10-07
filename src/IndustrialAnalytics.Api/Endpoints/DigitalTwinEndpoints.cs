using IndustrialAnalytics.Contracts.DigitalTwins;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;

namespace IndustrialAnalytics.Api.Endpoints
{
    /// <summary>
    /// Read APIs over the ML-derived Digital Twin state produced by the
    /// Spark vibration inference pipeline.
    ///
    /// Routes follow the existing convention: declared without the
    /// "/api/v1" prefix, which Program.cs supplies via MapGroup.
    ///
    /// The responses deliberately separate MODEL OUTPUT (anomaly,
    /// classification) from inputs (operatingContext, features) and from
    /// demoGroundTruth, which is demonstration metadata only. Raw
    /// 2048-sample vibration arrays are never exposed here - only the
    /// window's sequence range and its engineered features.
    /// </summary>
    public static class DigitalTwinEndpoints
    {
        private const int DefaultHistoryTake = 60;
        private const int MaxHistoryTake = 500;

        public static IEndpointRouteBuilder MapDigitalTwins(this IEndpointRouteBuilder app)
        {
            app.MapGet("/digital-twins",
                async (bool? includeGroundTruth, IDigitalTwinQueryRepository repo, CancellationToken ct) =>
                {
                    var items = await repo.GetAllAsync(includeGroundTruth ?? true, ct);
                    return Results.Ok(new DigitalTwinListResponse(items.Count, items));
                })
                .WithOpenApi(op =>
                {
                    op.Summary = "Current ML Digital Twin state for every asset";
                    op.Description =
                        "One row per asset, derived from the most recent completed " +
                        "2048-sample vibration window. 'anomaly' and 'classification' " +
                        "are model output; 'demoGroundTruth' is simulated-scenario " +
                        "metadata and is never a model input. Set includeGroundTruth=false " +
                        "to omit it.";
                    return op;
                });

            app.MapGet("/digital-twins/{assetId}",
                async (string assetId, bool? includeGroundTruth, IDigitalTwinQueryRepository repo, CancellationToken ct) =>
                {
                    var state = await repo.GetByAssetAsync(assetId, includeGroundTruth ?? true, ct);
                    return state is null
                        ? Results.NotFound(new { error = $"no Digital Twin state for asset '{assetId}'" })
                        : Results.Ok(state);
                })
                .WithOpenApi(op =>
                {
                    op.Summary = "Current ML Digital Twin state for one asset";
                    return op;
                });

            app.MapGet("/digital-twins/{assetId}/history",
                async (string assetId, int? take, bool? includeGroundTruth,
                       IDigitalTwinQueryRepository repo, CancellationToken ct) =>
                {
                    var limit = take is null
                        ? DefaultHistoryTake
                        : Math.Clamp(take.Value, 1, MaxHistoryTake);

                    var items = await repo.GetHistoryAsync(
                        assetId, limit, includeGroundTruth ?? true, ct);

                    return Results.Ok(new DigitalTwinHistoryResponse(assetId, items.Count, items));
                })
                .WithOpenApi(op =>
                {
                    op.Summary = "Recent inference history for an asset";
                    op.Description =
                        "Newest window first. Useful for showing that a classifier " +
                        "disagrees with ground truth on some windows while the anomaly " +
                        "detector still flags them.";
                    return op;
                });

            return app;
        }
    }
}

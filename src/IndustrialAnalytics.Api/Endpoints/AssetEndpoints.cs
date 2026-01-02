using IndustrialAnalytics.Infrastructure.Sql.Repositories;

namespace IndustrialAnalytics.Api.Endpoints
{
    public static class AssetsEndpoints
    {
        public static IEndpointRouteBuilder MapAssets(this IEndpointRouteBuilder app)
        {
            var g = app.MapGroup("/assets");

            g.MapGet("", async (string? q, int? limit, string? cursor, IAssetQueryRepository repo, CancellationToken ct) =>
            {
                var lim = Math.Clamp(limit ?? 100, 1, 500);
                return Results.Ok(await repo.GetAssetsAsync(q, lim, cursor, ct));
            });

            g.MapGet("/{assetId}/summary", async (string assetId, IAssetQueryRepository repo, CancellationToken ct) =>
            {
                var summary = await repo.GetAssetSummaryAsync(assetId, ct);
                return summary is null ? Results.NotFound() : Results.Ok(summary);
            });

            return app;
        }
    }
}

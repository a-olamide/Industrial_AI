using IndustrialAnalytics.Infrastructure.Sql.Repositories;

namespace IndustrialAnalytics.Api.Endpoints
{
    public static class AssetsRecommendationsEndpoints
    {
        public static IEndpointRouteBuilder MapAssetRecommendations(this IEndpointRouteBuilder app)
        {
            app.MapGet("/assets/{assetId}/recommendations",
                async (string assetId, string? status, int? take, IRecommendationQueryRepository repo, CancellationToken ct) =>
                {
                    // normalize inputs
                    var s = string.IsNullOrWhiteSpace(status) ? null : status.Trim().ToUpperInvariant();
                    var t = Math.Clamp(take ?? 50, 1, 500);

                    // Optional: allow ACTIVE (OPEN+ACKED)
                    if (s is "ACTIVE")
                    {
                        // If you want ACTIVE, easiest is return both and filter in SQL later.
                        // For now we’ll treat ACTIVE as null and filter in repo later if you want.
                        // Better: extend repo to accept multiple statuses.
                        s = null;
                    }

                    var items = await repo.GetForAssetAsync(assetId, s, t, ct);
                    return Results.Ok(new { assetId, items });
                })
                .WithName("GetAssetRecommendations")
                .WithOpenApi(op =>
                {
                    op.Summary = "List recommendations for an asset";
                    op.Description = "Returns recommendations for the given asset. Optional status filter (OPEN, ACKED, CLOSED).";
                    return op;
                });

            return app;
        }
    }
}

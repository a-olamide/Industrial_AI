using IndustrialAnalytics.Infrastructure.Sql.Repositories;

namespace IndustrialAnalytics.Api.Endpoints
{
    public static class RiskEndpoints
    {
        public static IEndpointRouteBuilder MapRisk(this IEndpointRouteBuilder app)
        {
            app.MapGet("/assets/{assetId}/risk",
                async (string assetId, DateTimeOffset from, DateTimeOffset to, int? stepMinutes, IRiskQueryRepository repo, CancellationToken ct) =>
                {
                    if (to <= from) return Results.BadRequest(new { error = "to must be after from" });

                    var step = stepMinutes ?? 1;
                    if (step is not (1 or 5 or 15)) return Results.BadRequest(new { error = "stepMinutes must be 1,5,or 15" });

                    var resp = await repo.GetRiskSeriesAsync(assetId, from, to, step, ct);
                    return Results.Ok(resp);
                }).WithName("GetAssetRiskSeries")
                .WithOpenApi(op =>
                {
                    op.Summary = "Risk score time series";
                    op.Description = "Returns risk score points for the asset between [from,to) in UTC. stepMinutes supports 1,5,15.";
                    return op;
                });

            return app;
        }
    }
}

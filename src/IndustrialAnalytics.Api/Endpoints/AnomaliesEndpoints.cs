using IndustrialAnalytics.Infrastructure.Sql.Repositories;

namespace IndustrialAnalytics.Api.Endpoints
{
    public static class AnomaliesEndpoints
    {
        public static IEndpointRouteBuilder MapAnomalies(this IEndpointRouteBuilder app)
        {
            // note: NO "/api/v1" here
            app.MapGet("/assets/{assetId}/anomalies",
                async (string assetId, DateTime from, DateTime to, int? take, IAnomalyQueryRepository repo, CancellationToken ct) =>
                {
                    if (to <= from) return Results.BadRequest(new { error = "to must be after from" });

                    var limit = take is null ? 500 : Math.Clamp(take.Value, 1, 2000);

                    // Treat incoming DateTime as UTC. (Swagger "Z" input is best)
                    var fromUtc = DateTime.SpecifyKind(from, DateTimeKind.Utc);
                    var toUtc = DateTime.SpecifyKind(to, DateTimeKind.Utc);

                    var items = await repo.GetForAssetAsync(assetId, fromUtc, toUtc, limit, ct);
                    return Results.Ok(new { assetId, from = fromUtc, to = toUtc, items });
                })
                .WithOpenApi(op =>
                {
                    op.Summary = "Anomaly events in a time range";
                    op.Description = "Returns anomaly events for the asset between [from,to) in UTC.";
                    return op;
                });

            return app;
        }
    }
}

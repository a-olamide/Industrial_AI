using IndustrialAnalytics.Api.Services;

namespace IndustrialAnalytics.Api.Endpoints
{
    public static class InsightsEndpoints
    {
        public static IEndpointRouteBuilder MapInsights(this IEndpointRouteBuilder app)
        {
            app.MapGet("/assets/{assetId}/insight",
                async (string assetId, DateTime from, DateTime to, int? take, AssetInsightService svc, CancellationToken ct) =>
                {
                    if (to <= from) return Results.BadRequest(new { error = "to must be after from" });

                    var dto = await svc.BuildInsightAsync(assetId,
                        DateTime.SpecifyKind(from, DateTimeKind.Utc),
                        DateTime.SpecifyKind(to, DateTimeKind.Utc),
                        take ?? 5,
                        ct);

                    return Results.Ok(dto);
                })
                .WithOpenApi(op =>
                {
                    op.Summary = "AI insight summary";
                    op.Description = "Generates a concise natural-language insight using deterministic facts (risk/anomalies/recommendations) and an LLM.";
                    return op;
                });

            return app;
        }
    }
}

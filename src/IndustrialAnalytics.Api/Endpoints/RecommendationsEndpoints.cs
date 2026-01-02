using IndustrialAnalytics.Contracts.Recommendations;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;

namespace IndustrialAnalytics.Api.Endpoints
{
    public static class RecommendationsEndpoints
    {
        public static IEndpointRouteBuilder MapRecommendations(this IEndpointRouteBuilder app)
        {
            var g = app.MapGroup("/recommendations");

            g.MapGet("", async (
            string status,
            int? take,
            string? assetId,
            IRecommendationQueryRepository repo,
            CancellationToken ct) =>
            {
                var t = Math.Clamp(take ?? 100, 1, 500);
                status = string.IsNullOrWhiteSpace(status) ? "OPEN" : status.ToUpperInvariant();

                var items = await repo.GetQueueAsync(status, t, assetId, ct);

                return Results.Ok(new
                {
                    status,
                    take = t,
                    assetId,
                    items
                });
            });

            g.MapPost("/{id:long}/ack", async (long id, AckRecommendationRequest req, IRecommendationCommandRepository repo, CancellationToken ct) =>
            {
                if (string.IsNullOrWhiteSpace(req.By)) return Results.BadRequest(new { error = "by is required" });

                var ok = await repo.AckAsync(id, req.AckUntil, req.By, req.Note, ct);
                return ok ? Results.Ok(new { recommendationId = id, status = "ACKED" }) : Results.NotFound();
            });

            g.MapPost("/{id:long}/close", async (long id, CloseRecommendationRequest req, IRecommendationCommandRepository repo, CancellationToken ct) =>
            {
                if (string.IsNullOrWhiteSpace(req.By)) return Results.BadRequest(new { error = "by is required" });
                if (string.IsNullOrWhiteSpace(req.Reason)) return Results.BadRequest(new { error = "reason is required" });

                var ok = await repo.CloseAsync(id, req.Reason, req.By, ct);
                return ok ? Results.Ok(new { recommendationId = id, status = "CLOSED" }) : Results.NotFound();
            });

            return app;
        }
    }
}

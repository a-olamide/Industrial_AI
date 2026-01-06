using IndustrialAnalytics.Contracts.Anomalies;
using IndustrialAnalytics.Contracts.Assets;
using IndustrialAnalytics.Contracts.Insights;
using IndustrialAnalytics.Contracts.Recommendations;
using IndustrialAnalytics.Contracts.Risk;

namespace IndustrialAnalytics.Ui.Services
{
    public sealed class InsightsApiClient(HttpClient http)
    {
        public Task<AssetListResponse?> GetAssetsAsync(CancellationToken ct = default)
            => http.GetFromJsonAsync<AssetListResponse>("/api/v1/assets", ct);

        public Task<AssetSummaryDto?> GetAssetSummaryAsync(string assetId, CancellationToken ct = default)
            => http.GetFromJsonAsync<AssetSummaryDto>($"/api/v1/assets/{assetId}/summary", ct);

        public async Task<IReadOnlyList<RiskPointDto>?> GetRiskSeriesAsync(
            string assetId, DateTime fromUtc, DateTime toUtc, int stepMinutes = 1, CancellationToken ct = default)
        {
            // ISO 8601 works great: 2025-01-01T12:00:00Z
            var from = Uri.EscapeDataString(fromUtc.ToString("O"));
            var to = Uri.EscapeDataString(toUtc.ToString("O"));

            var resp = await http.GetFromJsonAsync<RiskSeriesResponse>(
            $"/api/v1/assets/{assetId}/risk?from={from}&to={to}&stepMinutes={stepMinutes}", ct);

            return resp?.Points ?? [];
        }

        public Task<AssetRecommendationsResponseDto?> GetRecommendationsAsync(
            string assetId, string status = "OPEN", int take = 50, CancellationToken ct = default)
            => http.GetFromJsonAsync<AssetRecommendationsResponseDto>(
                $"/api/v1/assets/{assetId}/recommendations?status={status}&take={take}", ct);

        public async Task<bool> AckRecommendationAsync(long id, DateTime? ackUntil, string by, string? note, CancellationToken ct = default)
        {
            var req = new AckRecommendationRequest(ackUntil, by, note);

            var resp = await http.PostAsJsonAsync($"/api/v1/recommendations/{id}/ack", req, ct);
            return resp.IsSuccessStatusCode;
        }

        public async Task<bool> CloseRecommendationAsync(long id, string reason, string by, CancellationToken ct = default)
        {
            var req = new CloseRecommendationRequest(reason, by);

            var resp = await http.PostAsJsonAsync($"/api/v1/recommendations/{id}/close", req, ct);
            return resp.IsSuccessStatusCode;
        }
        public Task<RecommendationsQueueResponseDto?> GetRecommendationsQueueAsync(
    string status = "OPEN",
    int take = 100,
    string? assetId = null,
    CancellationToken ct = default)
        {
            var qs = new List<string>
    {
        $"status={Uri.EscapeDataString(status)}",
        $"take={take}"
    };

            if (!string.IsNullOrWhiteSpace(assetId))
                qs.Add($"assetId={Uri.EscapeDataString(assetId)}");

            return http.GetFromJsonAsync<RecommendationsQueueResponseDto>(
                $"/api/v1/recommendations?{string.Join("&", qs)}", ct);
        }

        public Task<AnomalyListResponseDto?> GetAnomaliesAsync(
    string assetId, DateTime fromUtc, DateTime toUtc, CancellationToken ct = default)
        {
            var from = Uri.EscapeDataString(fromUtc.ToString("O"));
            var to = Uri.EscapeDataString(toUtc.ToString("O"));

            return http.GetFromJsonAsync<AnomalyListResponseDto>(
                $"/api/v1/assets/{assetId}/anomalies?from={from}&to={to}", ct);
        }
        public async Task<AssetInsightDto?> GetInsightAsync(string assetId, DateTime fromUtc, DateTime toUtc, int take = 5, CancellationToken ct = default)
        {
            var from = Uri.EscapeDataString(fromUtc.ToString("O"));
            var to = Uri.EscapeDataString(toUtc.ToString("O"));

            using var req = new HttpRequestMessage(HttpMethod.Get,
                $"/api/v1/assets/{assetId}/insight?from={from}&to={to}&take={take}");

            using var resp = await http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
            if (!resp.IsSuccessStatusCode) return null;

            return await resp.Content.ReadFromJsonAsync<AssetInsightDto>(cancellationToken: ct);
        }
    }
}

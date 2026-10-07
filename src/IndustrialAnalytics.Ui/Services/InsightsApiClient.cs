using IndustrialAnalytics.Contracts.Anomalies;
using IndustrialAnalytics.Contracts.Assets;
using IndustrialAnalytics.Contracts.DigitalTwins;
using IndustrialAnalytics.Contracts.Insights;
using IndustrialAnalytics.Contracts.Recommendations;
using IndustrialAnalytics.Contracts.Risk;

namespace IndustrialAnalytics.Ui.Services
{
    public sealed class InsightsApiClient(HttpClient http)
    {
        public Task<AssetListResponse?> GetAssetsAsync(CancellationToken ct = default)
            => http.GetFromJsonAsync<AssetListResponse>("/api/v1/assets", ct);

        // ── ML Digital Twin (Spark streaming inference) ──────────────────
        public Task<DigitalTwinListResponse?> GetDigitalTwinsAsync(
            bool includeGroundTruth = true, CancellationToken ct = default)
            => http.GetFromJsonAsync<DigitalTwinListResponse>(
                $"/api/v1/digital-twins?includeGroundTruth={includeGroundTruth.ToString().ToLowerInvariant()}", ct);

        public Task<DigitalTwinStateDto?> GetDigitalTwinAsync(
            string assetId, bool includeGroundTruth = true, CancellationToken ct = default)
            => http.GetFromJsonAsync<DigitalTwinStateDto>(
                $"/api/v1/digital-twins/{assetId}?includeGroundTruth={includeGroundTruth.ToString().ToLowerInvariant()}", ct);

        public Task<DigitalTwinHistoryResponse?> GetDigitalTwinHistoryAsync(
            string assetId, int take = 60, CancellationToken ct = default)
            => http.GetFromJsonAsync<DigitalTwinHistoryResponse>(
                $"/api/v1/digital-twins/{assetId}/history?take={take}", ct);

        /// <summary>
        /// Requests an AI explanation of the asset's current ML result.
        /// <para>
        /// Only the asset id crosses the wire — the server owns the
        /// authoritative Digital Twin state and builds the Claude payload
        /// itself, so the browser can neither fabricate ML values nor see
        /// the API key.
        /// </para>
        /// Returns the explanation, or an error message for the UI.
        /// </summary>
        public async Task<(MaintenanceExplanationResponseDto? Result, string? Error)>
            GenerateTwinExplanationAsync(string assetId, bool refresh = false, CancellationToken ct = default)
        {
            using var resp = await http.PostAsync(
                $"/api/v1/digital-twins/{assetId}/explanation?refresh={refresh.ToString().ToLowerInvariant()}",
                content: null, ct);

            if (resp.IsSuccessStatusCode)
            {
                var ok = await resp.Content.ReadFromJsonAsync<MaintenanceExplanationResponseDto>(ct);
                return (ok, ok is null ? "Empty response from the API." : null);
            }

            var detail = await TryReadProblemDetailAsync(resp, ct);
            return (null, detail ?? $"Request failed ({(int)resp.StatusCode}).");
        }

        private static async Task<string?> TryReadProblemDetailAsync(
            HttpResponseMessage resp, CancellationToken ct)
        {
            try
            {
                using var doc = System.Text.Json.JsonDocument.Parse(
                    await resp.Content.ReadAsStringAsync(ct));
                if (doc.RootElement.TryGetProperty("detail", out var detail))
                    return detail.GetString();
                if (doc.RootElement.TryGetProperty("error", out var error))
                    return error.GetString();
            }
            catch (System.Text.Json.JsonException)
            {
                // Non-JSON error body; fall back to the status code.
            }
            return null;
        }

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

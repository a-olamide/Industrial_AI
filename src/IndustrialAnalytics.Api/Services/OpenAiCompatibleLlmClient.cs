using Microsoft.Extensions.Options;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace IndustrialAnalytics.Api.Services
{
    public sealed class OpenAiCompatibleLlmClient(HttpClient http, IOptions<LlmOptions> opt) : ILlmClient
    {
        public async Task<string> CompleteJsonAsync(string systemPrompt, string userJson, CancellationToken ct)
        {
            var o = opt.Value;

            http.BaseAddress = new Uri(o.BaseUrl);
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", o.ApiKey);

            var payload = new
            {
                model = o.Model,
                temperature = 0.2,
                response_format = new { type = "json_object" }, // many gateways support this
                messages = new object[]
                {
                new { role = "system", content = systemPrompt },
                new { role = "user", content = userJson }
                }
            };

            var json = JsonSerializer.Serialize(payload);
            using var resp = await http.PostAsync(
                "chat/completions",
                new StringContent(json, Encoding.UTF8, "application/json"),
                ct);

            var body = await resp.Content.ReadAsStringAsync(ct);
            resp.EnsureSuccessStatusCode();

            using var doc = JsonDocument.Parse(body);
            var content = doc.RootElement
                .GetProperty("choices")[0]
                .GetProperty("message")
                .GetProperty("content")
                .GetString();

            return content ?? "{}";
        }
    }
}

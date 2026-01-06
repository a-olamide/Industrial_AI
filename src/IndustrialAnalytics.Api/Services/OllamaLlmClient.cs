using System.Text.Json;
using System.Text.Json.Serialization;

namespace IndustrialAnalytics.Api.Services
{
    public sealed class OllamaLlmClient(HttpClient http) : ILlmClient
    {
        private static readonly JsonSerializerOptions JsonOpts = new()
        {
            PropertyNameCaseInsensitive = true
        };

        public async Task<string> CompleteJsonAsync(string systemPrompt, string userJson, CancellationToken ct)
        {
            const string model = "llama3.2"; // use a model you have in /api/tags

            var req = new OllamaChatRequest
            {
                Model = model,
                Stream = false,
                Format = "json",
                Messages =
                [
                    new OllamaMessage("system", systemPrompt),
                new OllamaMessage("user", userJson),
            ],
                Options = new OllamaOptions
                {
                    Temperature = 0.2,
                    NumPredict = 400,   // caps output length (important)
                    NumCtx = 2048       // optional: context window (keeps it snappy)
                }
            };
            var str= System.Text.Json.JsonSerializer.Serialize(req);
            using var resp = await http.PostAsJsonAsync("/api/chat", req, ct);
            resp.EnsureSuccessStatusCode();

            var payload = await resp.Content.ReadFromJsonAsync<OllamaChatResponse>(JsonOpts, ct);
            var content = payload?.Message?.Content?.Trim();

            return string.IsNullOrWhiteSpace(content)
                ? """{"headline":"Insight unavailable","summary":"Empty model response.","what_changed":[],"likely_causes":[],"recommended_next_steps":[],"confidence":0,"evidence_used":["ollama_empty"],"disclaimer":"Verify against underlying data."}"""
                : content;
        }

        private sealed record OllamaChatRequest
        {
            [JsonPropertyName("model")] public string Model { get; init; } = "";
            [JsonPropertyName("stream")] public bool Stream { get; init; } = false;
            [JsonPropertyName("format")] public string? Format { get; init; } = "json";
            [JsonPropertyName("messages")] public List<OllamaMessage> Messages { get; init; } = [];
            [JsonPropertyName("options")] public OllamaOptions? Options { get; init; }
        }

        private sealed record OllamaMessage(
            [property: JsonPropertyName("role")] string Role,
            [property: JsonPropertyName("content")] string Content,
            [property: JsonPropertyName("thinking")] string? Thinking = null);

        // ✅ This is where OllamaOptions goes (inside the request JSON as "options")
        private sealed record OllamaOptions
        {
            [JsonPropertyName("temperature")] public double Temperature { get; init; } = 0.2;
            [JsonPropertyName("num_predict")] public int NumPredict { get; init; } = 350;
            [JsonPropertyName("num_ctx")] public int? NumCtx { get; init; } = 2048;
        }

        private sealed record OllamaChatResponse
        {
            [JsonPropertyName("message")] public OllamaMessage? Message { get; init; }
        }
    }
}

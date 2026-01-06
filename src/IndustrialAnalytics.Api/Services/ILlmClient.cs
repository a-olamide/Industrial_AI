namespace IndustrialAnalytics.Api.Services
{
    public interface ILlmClient
    {
        Task<string> CompleteJsonAsync(string systemPrompt, string userJson, CancellationToken ct);
    }
}

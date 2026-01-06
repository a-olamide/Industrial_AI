namespace IndustrialAnalytics.Api.Services
{
    public sealed class LlmOptions
    {
        public string BaseUrl { get; set; } = "";
        public string ApiKey { get; set; } = "";
        public string Model { get; set; } = "";
    }
}

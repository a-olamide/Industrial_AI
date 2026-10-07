namespace IndustrialAnalytics.Api.Services
{
    /// <summary>
    /// Configuration for the Claude maintenance-explanation integration.
    ///
    /// <para>
    /// <b>The API key is never stored in appsettings.json.</b> It is read
    /// from the <c>ANTHROPIC_API_KEY</c> environment variable (or .NET user
    /// secrets in development). Everything else — model, token cap, timeout
    /// — is ordinary non-secret configuration under the <c>Claude</c>
    /// section and is safe to commit.
    /// </para>
    /// </summary>
    public sealed class ClaudeOptions
    {
        public const string SectionName = "Claude";

        /// <summary>
        /// Environment variable holding the API key. Named here rather than
        /// inlined so the lookup is greppable and testable.
        /// </summary>
        public const string ApiKeyEnvironmentVariable = "ANTHROPIC_API_KEY";

        /// <summary>
        /// Default model. Claude Haiku 4.5 is chosen deliberately: this
        /// workload is a short, schema-constrained explanation of a handful
        /// of numbers rendered on a dashboard, which is exactly the shape
        /// Haiku handles well, and it is the cheapest current model at
        /// $1/$5 per MTok. Override with <c>Claude:Model</c> (e.g.
        /// <c>claude-opus-5</c>) if explanation quality matters more than
        /// cost for a given deployment.
        /// </summary>
        public string Model { get; set; } = "claude-haiku-4-5";

        /// <summary>
        /// Output cap. The structured schema is five short fields; 1024
        /// tokens is comfortably more than a dashboard-sized answer needs
        /// and bounds the worst-case cost of a single request.
        /// </summary>
        public int MaxTokens { get; set; } = 1024;

        /// <summary>Per-request timeout. Dashboards must not hang.</summary>
        public int TimeoutSeconds { get; set; } = 30;

        /// <summary>
        /// How many recent history windows to summarise into the trend
        /// block. Counts only, so this costs a few tokens regardless.
        /// </summary>
        public int TrendWindowCount { get; set; } = 60;

        /// <summary>
        /// Cache generated explanations against
        /// (assetId, windowEndSequence) so repeated requests for an
        /// unchanged Digital Twin state do not re-bill the API.
        /// </summary>
        public bool CacheEnabled { get; set; } = true;

        public int CacheMinutes { get; set; } = 30;

        /// <summary>
        /// Resolved API key, or null when unconfigured. Environment variable
        /// wins; configuration is the fallback so user-secrets also works.
        /// </summary>
        public string? ApiKey { get; set; }

        public bool IsConfigured => !string.IsNullOrWhiteSpace(ApiKey);
    }
}

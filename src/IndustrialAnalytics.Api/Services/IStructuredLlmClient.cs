using System.Text.Json;

namespace IndustrialAnalytics.Api.Services
{
    /// <summary>
    /// Extends the existing <see cref="ILlmClient"/> family with a
    /// schema-constrained completion.
    ///
    /// <para>
    /// <see cref="ILlmClient.CompleteJsonAsync"/> asks a provider for "some
    /// JSON" and hopes. That was adequate for the Ollama insight card, but
    /// the Digital Twin explanation is rendered into fixed UI fields, so the
    /// shape has to be guaranteed rather than requested. This adds exactly
    /// one method for that and changes nothing about the existing interface
    /// or its two implementations.
    /// </para>
    /// </summary>
    public interface IStructuredLlmClient : ILlmClient
    {
        /// <summary>Model identifier in use, for logging and response metadata.</summary>
        string ModelName { get; }

        /// <summary>True when credentials are present and a call can be attempted.</summary>
        bool IsConfigured { get; }

        /// <summary>
        /// Complete against a JSON Schema the provider enforces.
        /// </summary>
        /// <param name="systemPrompt">Constraints and role.</param>
        /// <param name="userJson">The serialized evidence bundle.</param>
        /// <param name="jsonSchema">Schema the response must satisfy.</param>
        Task<string> CompleteStructuredAsync(
            string systemPrompt,
            string userJson,
            JsonElement jsonSchema,
            CancellationToken ct);
    }
}

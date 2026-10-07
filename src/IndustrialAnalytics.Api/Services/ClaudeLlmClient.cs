using System.Text.Json;
using Anthropic;
using Anthropic.Models.Messages;
using Microsoft.Extensions.Options;

namespace IndustrialAnalytics.Api.Services
{
    /// <summary>
    /// Claude-backed implementation of the existing LLM abstraction, using
    /// the official Anthropic C# SDK.
    ///
    /// <para>
    /// Registered alongside — not instead of — the Ollama client, so the
    /// existing insight card keeps its provider. Only the Digital Twin
    /// explanation resolves this one.
    /// </para>
    ///
    /// <para>
    /// Security boundary: this type only ever runs server-side. The API key
    /// is read from the environment at construction and is never written to
    /// a log, a response DTO, or anything reachable by the Blazor client.
    /// </para>
    /// </summary>
    public sealed class ClaudeLlmClient : IStructuredLlmClient
    {
        private readonly ClaudeOptions _options;
        private readonly ILogger<ClaudeLlmClient> _logger;
        private readonly AnthropicClient? _client;

        public ClaudeLlmClient(IOptions<ClaudeOptions> options, ILogger<ClaudeLlmClient> logger)
        {
            _options = options.Value;
            _logger = logger;

            if (_options.IsConfigured)
            {
                _client = new AnthropicClient { ApiKey = _options.ApiKey };
                // Log that a key was found — never any part of the key itself.
                _logger.LogInformation(
                    "Claude explanation client ready (model {Model}, maxTokens {MaxTokens}).",
                    _options.Model, _options.MaxTokens);
            }
            else
            {
                _logger.LogWarning(
                    "Claude explanation client disabled: {EnvVar} is not set.",
                    ClaudeOptions.ApiKeyEnvironmentVariable);
            }
        }

        public string ModelName => _options.Model;

        public bool IsConfigured => _options.IsConfigured && _client is not null;

        /// <summary>
        /// Unconstrained completion, present so this type satisfies the
        /// existing <see cref="ILlmClient"/> contract. The explanation path
        /// uses <see cref="CompleteStructuredAsync"/>.
        /// </summary>
        public Task<string> CompleteJsonAsync(string systemPrompt, string userJson, CancellationToken ct)
            => CompleteStructuredAsync(systemPrompt, userJson, default, ct);

        public async Task<string> CompleteStructuredAsync(
            string systemPrompt,
            string userJson,
            JsonElement jsonSchema,
            CancellationToken ct)
        {
            if (_client is null)
            {
                throw new ClaudeNotConfiguredException(
                    $"{ClaudeOptions.ApiKeyEnvironmentVariable} is not configured on the server.");
            }

            // Schema-constrained output: the provider enforces the shape, so
            // the UI's five fields are guaranteed rather than hoped for.
            // OutputConfig is init-only, so it is built before the params.
            OutputConfig? outputConfig = jsonSchema.ValueKind == JsonValueKind.Object
                ? new OutputConfig
                {
                    Format = new JsonOutputFormat
                    {
                        Schema = BuildSchemaDictionary(jsonSchema),
                    },
                }
                : null;

            var parameters = new MessageCreateParams
            {
                Model = _options.Model,
                MaxTokens = _options.MaxTokens,
                System = systemPrompt,
                Messages = [new() { Role = Role.User, Content = userJson }],
                OutputConfig = outputConfig,
            };

            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeout.CancelAfter(TimeSpan.FromSeconds(_options.TimeoutSeconds));

            Message response;
            try
            {
                response = await _client.Messages.Create(parameters, cancellationToken: timeout.Token);
            }
            catch (OperationCanceledException) when (!ct.IsCancellationRequested)
            {
                throw new ClaudeUnavailableException(
                    $"Claude did not respond within {_options.TimeoutSeconds}s.");
            }
            catch (Exception ex) when (ex is not ClaudeNotConfiguredException)
            {
                // Never surface the raw exception to the client: provider
                // errors can echo request detail.
                _logger.LogError(ex, "Claude request failed.");
                throw new ClaudeUnavailableException("The Claude API request failed.");
            }

            if (response.StopReason == StopReason.Refusal)
            {
                throw new ClaudeUnavailableException(
                    "Claude declined to answer this request.");
            }

            var text = string.Concat(
                response.Content.Select(b => b.Value).OfType<TextBlock>().Select(b => b.Text));

            if (string.IsNullOrWhiteSpace(text))
            {
                throw new ClaudeUnavailableException("Claude returned an empty response.");
            }

            _logger.LogInformation(
                "Claude explanation generated (model {Model}, in {InTokens} tok, out {OutTokens} tok).",
                _options.Model, response.Usage?.InputTokens, response.Usage?.OutputTokens);

            return text;
        }

        private static Dictionary<string, JsonElement> BuildSchemaDictionary(JsonElement schema)
        {
            var map = new Dictionary<string, JsonElement>();
            foreach (var property in schema.EnumerateObject())
            {
                map[property.Name] = property.Value;
            }
            return map;
        }
    }

    /// <summary>Server is missing the Claude API key.</summary>
    public sealed class ClaudeNotConfiguredException(string message) : Exception(message);

    /// <summary>Claude was unreachable, timed out, refused, or errored.</summary>
    public sealed class ClaudeUnavailableException(string message) : Exception(message);
}

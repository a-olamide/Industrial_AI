using IndustrialAnalytics.Api.Endpoints;
using IndustrialAnalytics.Api.Services;
using IndustrialAnalytics.Infrastructure.DependencyInjection;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// DI
builder.Services.AddInfrastructure(builder.Configuration);

// Repos for API
builder.Services.AddScoped<IAssetQueryRepository, AssetQueryRepository>();
builder.Services.AddScoped<IRiskQueryRepository, RiskQueryRepository>();
builder.Services.AddScoped<IRecommendationCommandRepository, RecommendationCommandRepository>();
builder.Services.Configure<LlmOptions>(builder.Configuration.GetSection("Llm"));
//builder.Services.AddHttpClient<ILlmClient, OpenAiCompatibleLlmClient>();
builder.Services.AddHttpClient<ILlmClient, OllamaLlmClient>(c =>
{
    c.BaseAddress = new Uri("http://localhost:11434");
    c.Timeout = TimeSpan.FromMinutes(10);
});
builder.Services.AddScoped<AssetInsightService>();

// ── Claude maintenance explanations ───────────────────────────────────────
// The API key comes from the ANTHROPIC_API_KEY environment variable (or user
// secrets), never from appsettings.json. Everything else is plain config.
builder.Services.AddMemoryCache();
builder.Services.Configure<ClaudeOptions>(opts =>
{
    builder.Configuration.GetSection(ClaudeOptions.SectionName).Bind(opts);
    opts.ApiKey =
        Environment.GetEnvironmentVariable(ClaudeOptions.ApiKeyEnvironmentVariable)
        ?? builder.Configuration[$"{ClaudeOptions.SectionName}:ApiKey"];
});
// Registered as IStructuredLlmClient only, so the existing ILlmClient
// binding (Ollama, used by the legacy insight card) is left untouched.
builder.Services.AddSingleton<IStructuredLlmClient, ClaudeLlmClient>();
builder.Services.AddScoped<DigitalTwinExplanationService>();

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
//builder.Services.AddOpenApi();

var app = builder.Build();

// Configure the HTTP request pipeline.
//if (app.Environment.IsDevelopment())
//{
//    app.MapOpenApi();
//}
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "IndustrialAnalytics API v1");
        c.RoutePrefix = "swagger"; // default, but explicit
    });
}

app.UseHttpsRedirection();

var v1 = app.MapGroup("/api/v1")
    .WithTags("IndustrialAnalytics.Api");

v1.MapAssets();
v1.MapRisk();
v1.MapRecommendations();
v1.MapAssetRecommendations();
v1.MapAnomalies();
v1.MapInsights();
v1.MapDigitalTwins();

app.Run();


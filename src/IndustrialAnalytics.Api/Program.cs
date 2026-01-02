using IndustrialAnalytics.Api.Endpoints;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;
using IndustrialAnalytics.Infrastructure.DependencyInjection;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// DI
builder.Services.AddInfrastructure(builder.Configuration);

// Repos for API
builder.Services.AddScoped<IAssetQueryRepository, AssetQueryRepository>();
builder.Services.AddScoped<IRiskQueryRepository, RiskQueryRepository>();
builder.Services.AddScoped<IRecommendationCommandRepository, RecommendationCommandRepository>();


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

app.Run();


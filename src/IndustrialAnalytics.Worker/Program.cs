using IndustrialAnalytics.Domain.Anomalies;
using IndustrialAnalytics.Domain.DependencyInjection;
using IndustrialAnalytics.Domain.Risk;
using IndustrialAnalytics.Infrastructure.DependencyInjection;
using IndustrialAnalytics.Worker;
using IndustrialAnalytics.Worker.Options;
using IndustrialAnalytics.Worker.Workers;
using Serilog;

var builder = Host.CreateApplicationBuilder(args);
builder.Services.AddHostedService<AnomalyWorker>();
// Bind options
builder.Services.Configure<WorkerOptions>(builder.Configuration.GetSection("Worker"));
builder.Services.Configure<Thresholds>(builder.Configuration.GetSection("AnomalyDetection"));

builder.Services.AddHostedService<RiskWorker>();
builder.Services.Configure<RiskWorkerOptions>(builder.Configuration.GetSection("Risk"));
builder.Services.Configure<RiskOptions>(builder.Configuration.GetSection("RiskScoring"));

builder.Services.AddHostedService<RecommendationWorker>();

builder.Services.Configure<RecommendationWorkerOptions>(
    builder.Configuration.GetSection("Recommendations"));

builder.Services.Configure<IndustrialAnalytics.Domain.Recommendations.RecommendationOptions>(
    builder.Configuration.GetSection("RecommendationRules"));
// Add layers
builder.Services.AddDomain();
builder.Services.AddInfrastructure();

// Logging
builder.Services.AddSerilog(cfg => cfg.ReadFrom.Configuration(builder.Configuration).WriteTo.Console());


var host = builder.Build();
host.Run();

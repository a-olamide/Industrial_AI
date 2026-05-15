using IndustrialTelemetry.Producer.Services;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var host = Host.CreateDefaultBuilder(args)
    .ConfigureServices((ctx, services) =>
    {
        services.Configure<KafkaOptions>(ctx.Configuration.GetSection("Kafka"));
        services.Configure<TelemetryOptions>(ctx.Configuration.GetSection("Telemetry"));
        services.Configure<AnomalyOptions>(ctx.Configuration.GetSection("Anomaly"));
        services.AddSingleton<TelemetrySimulator>();
        services.AddHostedService<KafkaTelemetryProducer>();
    })
    .Build();

await host.RunAsync();

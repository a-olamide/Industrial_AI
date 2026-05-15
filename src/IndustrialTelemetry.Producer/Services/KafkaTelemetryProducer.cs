using System.Text.Json;
using Confluent.Kafka;
using IndustrialTelemetry.Producer.Models;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace IndustrialTelemetry.Producer.Services;

public sealed class KafkaOptions
{
    public string BootstrapServers { get; init; } = "localhost:9092";
    public string Topic { get; init; } = "industrial-telemetry";
    public string ClientId { get; init; } = "industrial-telemetry-producer";
    public int LingerMs { get; init; } = 5;
    public int BatchSize { get; init; } = 65536;
    public string Acks { get; init; } = "all";
}

public sealed class TelemetryOptions
{
    public int EmitIntervalMs { get; init; } = 1000;
}

public sealed class AnomalyOptions
{
    public double Probability { get; init; } = 0.05;
    public int DurationSeconds { get; init; } = 30;
}

public sealed class KafkaTelemetryProducer : BackgroundService
{
    private static readonly JsonSerializerOptions JsonOpts = new() { WriteIndented = false };

    private readonly ILogger<KafkaTelemetryProducer> _logger;
    private readonly TelemetrySimulator _simulator;
    private readonly KafkaOptions _kafka;
    private readonly TelemetryOptions _telemetry;

    public KafkaTelemetryProducer(
        ILogger<KafkaTelemetryProducer> logger,
        TelemetrySimulator simulator,
        IOptions<KafkaOptions> kafkaOptions,
        IOptions<TelemetryOptions> telemetryOptions)
    {
        _logger = logger;
        _simulator = simulator;
        _kafka = kafkaOptions.Value;
        _telemetry = telemetryOptions.Value;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var producerConfig = new ProducerConfig
        {
            BootstrapServers = _kafka.BootstrapServers,
            ClientId = _kafka.ClientId,
            LingerMs = _kafka.LingerMs,
            BatchSize = _kafka.BatchSize,
            MessageMaxBytes = 1_000_000,
            SocketKeepaliveEnable = true,
            Acks = _kafka.Acks switch
            {
                "all"    => Acks.All,
                "none"   => Acks.None,
                "leader" => Acks.Leader,
                _        => Acks.All
            }
        };

        using var producer = new ProducerBuilder<string, string>(producerConfig)
            .SetErrorHandler((_, e) =>
                _logger.LogError("Kafka producer error [{Code}]: {Reason}", e.Code, e.Reason))
            .Build();

        _logger.LogInformation(
            "Telemetry producer started → {Servers} | topic={Topic} | interval={Interval}ms | assets={AssetCount}",
            _kafka.BootstrapServers, _kafka.Topic, _telemetry.EmitIntervalMs, _simulator.Assets.Count);

        var interval = TimeSpan.FromMilliseconds(_telemetry.EmitIntervalMs);
        var cycleCount = 0L;

        while (!stoppingToken.IsCancellationRequested)
        {
            var cycleStart = DateTimeOffset.UtcNow;
            var produced = 0;

            try
            {
                foreach (var asset in _simulator.Assets)
                {
                    foreach (var evt in _simulator.GenerateEvents(asset))
                    {
                        var key = $"{evt.AssetId}:{evt.Tag}";
                        var payload = JsonSerializer.Serialize(evt, JsonOpts);

                        try
                        {
                            await producer.ProduceAsync(
                                _kafka.Topic,
                                new Message<string, string> { Key = key, Value = payload },
                                stoppingToken);

                            produced++;
                        }
                        catch (ProduceException<string, string> ex)
                        {
                            _logger.LogWarning(
                                "Delivery failed [{Code}] key={Key}: {Reason}",
                                ex.Error.Code, key, ex.Error.Reason);
                        }
                    }
                }

                cycleCount++;

                // Heartbeat every 60 s at Info; per-cycle detail at Debug
                if (cycleCount % 60 == 0)
                    _logger.LogInformation(
                        "Heartbeat | cycle={Cycle} | produced={Count} events this cycle",
                        cycleCount, produced);
                else
                    _logger.LogDebug(
                        "Cycle {Cycle} | {Count} events | {ElapsedMs:F1} ms",
                        cycleCount, produced, (DateTimeOffset.UtcNow - cycleStart).TotalMilliseconds);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unexpected error in telemetry cycle {Cycle}", cycleCount);
            }

            var delay = interval - (DateTimeOffset.UtcNow - cycleStart);
            if (delay > TimeSpan.Zero)
            {
                try { await Task.Delay(delay, stoppingToken); }
                catch (OperationCanceledException) { break; }
            }
        }

        var queued = producer.Flush(TimeSpan.FromSeconds(15));
        _logger.LogInformation(
            "Producer shut down after {Cycles} cycles. Messages left in queue: {Queued}",
            cycleCount, queued);
    }
}

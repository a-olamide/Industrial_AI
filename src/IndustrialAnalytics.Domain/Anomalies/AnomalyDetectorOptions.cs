using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Anomalies
{
    public sealed record AnomalyDetectorOptions
    {
        public int MinRunMinutes { get; init; } = 5;
        public int MinRowsForBaseline { get; init; } = 6;
        public int LookbackMinutesDev { get; init; } = 10;

        // Z-score thresholds
        public double VibSpikeZ { get; init; } = 3.0;
        public double VibSevereZ { get; init; } = 4.0;
        public double TempHighZ { get; init; } = 3.0;
        public double TempSevereZ { get; init; } = 4.0;
        public double CurrentHighZ { get; init; } = 3.0;
        public double CurrentSevereZ { get; init; } = 4.0;

        // Drift thresholds (slope-based)
        public double VibDriftSlope { get; init; } = 0.05;
        public double VibSevereSlope { get; init; } = 0.10;
        public double TempDriftSlope { get; init; } = 0.10;
        public double TempSevereSlope { get; init; } = 0.20;
        public double CurrentDriftSlope { get; init; } = 0.05;
        public double CurrentSevereSlope { get; init; } = 0.10;

        // Flow drop thresholds
        public double FlowDropPct { get; init; } = -0.10;
        public double FlowDropSeverePct { get; init; } = -0.20;

        // Epsilon for stddev stability
        public double VibStdEps { get; init; } = 0.05;
        public double TempStdEps { get; init; } = 0.50;
        public double CurrentStdEps { get; init; } = 0.20;
    }
}

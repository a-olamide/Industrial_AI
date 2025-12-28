using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Risk
{
    public sealed record RiskOptions
    {
        public int WindowMinutes { get; init; } = 30;

        public int LowMax { get; init; } = 29;
        public int MedMax { get; init; } = 69;

        // base points severity2/severity3
        public int TempDriftS2 { get; init; } = 12;
        public int TempDriftS3 { get; init; } = 20;
        public int VibDriftS2 { get; init; } = 15;
        public int VibDriftS3 { get; init; } = 25;
        public int CurrentDriftS2 { get; init; } = 8;
        public int CurrentDriftS3 { get; init; } = 14;
        public int VibSpikeS2 { get; init; } = 18;
        public int VibSpikeS3 { get; init; } = 30;
        public int FlowDropS2 { get; init; } = 12;
        public int FlowDropS3 { get; init; } = 20;
    }
}

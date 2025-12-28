using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Telemetry
{
    public static class Signals
    {
        // High-level signal names used in anomalies + UI
        public const string Vibration = "vibration";
        public const string Temperature = "temp";
        public const string Current = "current";
        public const string Flow = "flow";
        public const string Pressure = "pressure";
    }
}

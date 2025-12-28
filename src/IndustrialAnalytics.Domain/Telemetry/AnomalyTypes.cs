using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Telemetry
{
    public static class AnomalyTypes
    {
        public const string VibSpike = "VIB_SPIKE";
        public const string VibDrift = "VIB_DRIFT";
        public const string TempHigh = "TEMP_HIGH";
        public const string TempDrift = "TEMP_DRIFT";
        public const string CurrentHigh = "CURRENT_HIGH";
        public const string CurrentDrift = "CURRENT_DRIFT";
        public const string FlowDrop = "FLOW_DROP";
    }
}

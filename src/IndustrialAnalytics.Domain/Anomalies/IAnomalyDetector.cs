using IndustrialAnalytics.Domain.Telemetry;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Anomalies
{
    public interface IAnomalyDetector
    {
        IReadOnlyList<AnomalyEvent> Detect(
            FeatureSnapshot current,
            IReadOnlyList<FeatureSnapshot> lookback,
            Thresholds thresholds);
    }
}

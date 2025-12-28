using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Risk
{
    public interface IRiskEngine
    {
        RiskResult Compute(string assetId, DateTime minuteTs, IReadOnlyList<AnomalyRow> window, RiskOptions opt);
    }
}

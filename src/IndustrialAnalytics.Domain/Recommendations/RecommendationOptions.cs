using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Recommendations
{
    public sealed record RecommendationOptions
    {
        public int CooldownHoursHigh { get; init; } = 12;
        public int CooldownHoursMed { get; init; } = 24;
        public int CooldownHoursLow { get; init; } = 48;
    }
}

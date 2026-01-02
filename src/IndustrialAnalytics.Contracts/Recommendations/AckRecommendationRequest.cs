using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Recommendations
{
    public sealed record AckRecommendationRequest(
        DateTime? AckUntil,
        string By,
        string? Note
    );
}

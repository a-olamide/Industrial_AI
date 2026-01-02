using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Recommendations
{
    public sealed record CloseRecommendationRequest(
        string Reason,
        string By
    );
}

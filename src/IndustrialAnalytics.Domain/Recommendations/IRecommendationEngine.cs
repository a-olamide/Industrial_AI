using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Recommendations
{
    public interface IRecommendationEngine
    {
        IReadOnlyList<Recommendation> Generate(RiskSnapshot risk, RecommendationOptions opt);
    }
}

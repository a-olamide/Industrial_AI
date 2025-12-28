using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.DependencyInjection
{

    public static class ServiceCollectionExtensions
    {
        public static IServiceCollection AddDomain(this IServiceCollection services)
        {
            // Register domain services (pure logic)
            services.AddSingleton<Anomalies.IAnomalyDetector, Anomalies.HybridAnomalyDetector>();
            services.AddSingleton<IndustrialAnalytics.Domain.Risk.IRiskEngine, IndustrialAnalytics.Domain.Risk.RiskEngine>();
            services.AddSingleton<Recommendations.IRecommendationEngine, Recommendations.RecommendationEngine>();


            return services;
        }
    }
}

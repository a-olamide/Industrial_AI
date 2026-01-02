using IndustrialAnalytics.Infrastructure.Sql;
using IndustrialAnalytics.Infrastructure.Sql.Repositories;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.DependencyInjection
{
    public static class ServiceCollectionExtensions
    {
        public static IServiceCollection AddInfrastructure(this IServiceCollection services, IConfiguration cfg)
        {
            services.AddSingleton<ISqlConnectionFactory>(_ =>
                new SqlConnectionFactory(
                    cfg.GetConnectionString("Sql")
                    ?? throw new InvalidOperationException("Missing connection string: Sql")));

            services.AddSingleton<ICheckpointStore, CheckpointStore>();

            services.AddSingleton<IFeatureRepository, FeatureRepository>();
            services.AddSingleton<IAnomalyEventRepository, AnomalyEventRepository>();
            services.AddSingleton<IAnomalyQueryRepository, AnomalyQueryRepository>();
            services.AddSingleton<IRiskRepository, RiskRepository>();
            services.AddSingleton<IRiskQueryRepository, RiskQueryRepository>();
            services.AddSingleton<IRecommendationRepository, RecommendationRepository>();
            services.AddSingleton<IRecommendationCommandRepository, RecommendationCommandRepository>();
            services.AddSingleton<IRecommendationQueryRepository, RecommendationQueryRepository>();

            return services;
        }
    }
}

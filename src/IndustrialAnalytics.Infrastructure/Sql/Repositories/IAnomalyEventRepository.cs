using IndustrialAnalytics.Domain.Anomalies;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public interface IAnomalyEventRepository
    {
        Task InsertManyAsync(IEnumerable<AnomalyEvent> events, CancellationToken ct);
    }
}

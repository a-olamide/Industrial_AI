using IndustrialAnalytics.Domain.Risk;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public interface IRiskRepository
    {
        Task UpsertMinuteAsync(RiskResult r, CancellationToken ct);
        Task UpsertCurrentAsync(RiskResult r, CancellationToken ct);
    }
}

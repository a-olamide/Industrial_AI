using IndustrialAnalytics.Domain.Recommendations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public interface IRiskQueryRepository
    {
        Task<IReadOnlyList<RiskSnapshot>> GetCurrentAsync(CancellationToken ct);
    }
}

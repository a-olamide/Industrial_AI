using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql
{
    public interface ICheckpointStore
    {
        Task<DateTime?> GetLastAsync(string workerName, CancellationToken ct);
        Task SetLastAsync(string workerName, DateTime lastMinuteTs, CancellationToken ct);
    }
}

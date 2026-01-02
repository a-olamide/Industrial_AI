using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public interface IRecommendationCommandRepository
    {
        Task<bool> AckAsync(long id, DateTime? ackUntil, string by, string? note, CancellationToken ct);
        Task<bool> CloseAsync(long id, string reason, string by, CancellationToken ct);
    }
}

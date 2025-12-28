using Dapper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql
{
    public class CheckpointStore(ISqlConnectionFactory f) : ICheckpointStore
    {
        public async Task<DateTime?> GetLastAsync(string workerName, CancellationToken ct)
        {
            using var conn = f.Create();
            return await conn.QuerySingleOrDefaultAsync<DateTime?>(
                new CommandDefinition(
                    "SELECT last_minute_ts FROM dbo.worker_checkpoint WHERE worker_name=@workerName",
                    new { workerName }, cancellationToken: ct));
        }

        public async Task SetLastAsync(string workerName, DateTime lastMinuteTs, CancellationToken ct)
        {
            using var conn = f.Create();
            await conn.ExecuteAsync(new CommandDefinition(@"
MERGE dbo.worker_checkpoint AS tgt
USING (SELECT @workerName AS worker_name, @lastMinuteTs AS last_minute_ts) AS src
ON tgt.worker_name = src.worker_name
WHEN MATCHED THEN UPDATE SET last_minute_ts = src.last_minute_ts, updated_at = SYSUTCDATETIME()
WHEN NOT MATCHED THEN INSERT (worker_name, last_minute_ts) VALUES (src.worker_name, src.last_minute_ts);",
                new { workerName, lastMinuteTs }, cancellationToken: ct));
        }
    }
}

using Dapper;
using IndustrialAnalytics.Domain.Anomalies;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql.Repositories
{
    public sealed class AnomalyEventRepository(ISqlConnectionFactory f) : IAnomalyEventRepository
    {
        public async Task InsertManyAsync(IEnumerable<AnomalyEvent> events, CancellationToken ct)
        {
            var batch = events as AnomalyEvent[] ?? events.ToArray();
            if (batch.Length == 0) return;

            using var conn = f.Create();

            // ✅ Required when you want to begin a transaction
            if (conn is Microsoft.Data.SqlClient.SqlConnection sqlConn)
                await sqlConn.OpenAsync(ct);
            else
                conn.Open();

            using var tx = conn.BeginTransaction();

            const string sql = @"
                BEGIN TRY
                    INSERT INTO dbo.asset_anomaly_events
                    (asset_id, minute_ts, anomaly_type, signal, score, severity, reason, evidence_json)
                    VALUES
                    (@AssetId, @MinuteTs, @AnomalyType, @Signal, @Score, @Severity, @Reason, @EvidenceJson);
                END TRY
                BEGIN CATCH
                    IF ERROR_NUMBER() <> 2601 AND ERROR_NUMBER() <> 2627
                    THROW;
                END CATCH;";

            foreach (var e in batch)
            {
                await conn.ExecuteAsync(new CommandDefinition(
                    sql, e, transaction: tx, cancellationToken: ct));
            }

            tx.Commit();
        }
    }
}

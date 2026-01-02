using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Domain.Risk
{
    //    public sealed record AnomalyRow(
    //    string AssetId,
    //    DateTime MinuteTs,
    //    string AnomalyType,
    //    string Signal,
    //    double? Score,
    //    byte Severity
    //);

    public sealed class AnomalyRow
    {
        public long AnomalyId { get; set; }
        public string AssetId { get; set; } = "";
        public DateTime MinuteTs { get; set; }
        public string AnomalyType { get; set; } = "";
        public string Signal { get; set; } = "";
        public double Score { get; set; }
        public int Severity { get; set; }
        public string Reason { get; set; } = "";
        public string? Evidence { get; set; }
        public DateTime CreatedAt { get; set; }
    }
}

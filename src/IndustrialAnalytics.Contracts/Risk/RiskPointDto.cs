using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Risk
{
    public sealed class RiskPointDto
    {
        public DateTime MinuteTs { get; set; }
        public int Score { get; set; }
        public string? Level { get; set; }
        public string? FailureMode { get; set; }
    }
}

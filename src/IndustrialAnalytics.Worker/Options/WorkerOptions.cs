using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Worker.Options
{

    public sealed record WorkerOptions
    {
        public string Name { get; init; } = "anomaly-worker";
        public int PollSeconds { get; init; } = 10;
    }
}

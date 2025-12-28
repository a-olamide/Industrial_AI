using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Worker.Options
{
    public sealed record RecommendationWorkerOptions
    {
        public string WorkerName { get; init; } = "recommendation-worker";
        public int PollSeconds { get; init; } = 30;
    }
}

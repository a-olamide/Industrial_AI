using IndustrialAnalytics.Contracts.Dtos;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Anomalies
{
    public sealed record AnomalyListResponseDto(
        string AssetId,
        IReadOnlyList<AnomalyDto> Items
    );
}

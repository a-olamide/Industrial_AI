using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Contracts.Assets
{
    public sealed record AssetDto(
        string AssetId,
        string? AssetType,
        string? Site,
        string? DisplayName,
        DateTime? LastSeenMinuteTs
    );
}

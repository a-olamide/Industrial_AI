using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql
{
    public interface ISqlConnectionFactory
    {
        IDbConnection Create();
    }
}

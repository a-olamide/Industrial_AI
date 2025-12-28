using Microsoft.Data.SqlClient;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IndustrialAnalytics.Infrastructure.Sql
{
    public class SqlConnectionFactory(IConfiguration cfg) : ISqlConnectionFactory
    {
        public IDbConnection Create() => new SqlConnection(cfg.GetConnectionString("Sql"));
    }
}

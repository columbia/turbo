import sqlglot
from turbo.api import TurboQuery


class SQLTurboQuery(TurboQuery):
    def __init__(self, sql_query: str):
        self.sql_query = sql_query

    def get_filter_clause(self):
        return sql_get_filter_clause(self.sql_query)

    def get_aggregation_type(self):
        return sql_get_aggregation_type(self.sql_query)

    def get_data_view_id(self):
        return sql_get_data_view_id(self.sql_query)


def sql_get_filter_clause(sql_query):
    sql = sqlglot.parse_one(sql_query)
    if sql.find(sqlglot.exp.Group):
        raise ValueError("Turbo doesn't support Groupbys for now")

    where = sql.find(sqlglot.exp.Where)

    for node in where.dfs():
        if not node[0].key in {
            "where",
            "paren",
            "and",
            "or",
            "eq",
            "in",
            "column",
            "identifier",
            "literal",
        }:
            raise ValueError("Turbo supports limited syntax.")
    filter_clauses = {}
    eqs = list(where.find_all(sqlglot.exp.EQ))
    for eq in eqs:
        attr_name = eq.left.alias_or_name
        attr_value = eq.right.alias_or_name
        if attr_name not in filter_clauses:
            filter_clauses[attr_name] = [attr_value]
        else:
            filter_clauses[attr_name] += [attr_value]

    ins = list(where.find_all(sqlglot.exp.In))
    for in_ in ins:
        attr_name = in_.this.alias_or_name
        for expr in in_.expressions:
            attr_value = expr.alias_or_name

            if attr_name not in filter_clauses:
                filter_clauses[attr_name] = [attr_value]
            else:
                filter_clauses[attr_name] += [attr_value]
    return filter_clauses


def sql_get_aggregation_type(sql_query):
    sql = sqlglot.parse_one(sql_query)

    if len(sql.expressions) > 1:
        raise ValueError("Turbo supports only one expression in the select statement.")
    if sql.expressions[0].key != "count":
        raise ValueError("Turbo supports only counts.")
    return "count"


def sql_get_data_view_id(sql_query):
    sql = sqlglot.parse_one(sql_query)
    if sql.find(sqlglot.exp.Join) or sql.find(sqlglot.exp.Having):
        raise ValueError("Turbo doesn't support Joins/Havings etc")
    data_view_id = sql.find(sqlglot.exp.From).alias_or_name
    return data_view_id


def sql_drop_where(sql_query):
    sql = sqlglot.parse_one(sql_query)
    sql_without_where = sql.transform(
        lambda node: None if isinstance(node, sqlglot.exp.Where) else node
    )
    return sql_without_where.sql()


if __name__ == "__main__":
    query = """SELECT count(*) FROM table WHERE gender in (male, female)"""
    print(sql_get_filter_clause(query))
    print(sql_get_aggregation_type(query))
    print(sql_get_data_view_id(query))

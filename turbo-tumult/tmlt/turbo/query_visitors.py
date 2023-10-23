from tmlt.analytics.query_expr import QueryExpr, QueryExprVisitor
from tmlt.analytics.query_expr import DropInfinity as DropInfExpr
from tmlt.analytics.query_expr import DropNullAndNan, EnforceConstraint
from tmlt.analytics.query_expr import Filter as FilterExpr
from tmlt.analytics.query_expr import FlatMap as FlatMapExpr
from tmlt.analytics.query_expr import (
    GroupByBoundedAverage,
    GroupByBoundedSTDEV,
    GroupByBoundedSum,
    GroupByBoundedVariance,
    GroupByCount,
    GroupByCountDistinct,
    GroupByQuantile,
)
from tmlt.analytics.query_expr import JoinPrivate as JoinPrivateExpr
from tmlt.analytics.query_expr import JoinPublic as JoinPublicExpr
from tmlt.analytics.query_expr import Map as MapExpr
from tmlt.analytics.query_expr import QueryExpr, QueryExprVisitor
from tmlt.analytics.query_expr import Rename as RenameExpr
from tmlt.analytics.query_expr import ReplaceInfinity, ReplaceNullAndNan
from tmlt.analytics.query_expr import Select as SelectExpr

from typing import Any

import sqlglot


class QueryVisitor(QueryExprVisitor):
    """A visitor to Parse Turbo info from a query expression.
    Raises errors for all cases not supported by Turbo"""

    def __init__(
        self,
    ):
        """Constructor for a TurboVisitor."""
        pass

    def _visit_child(self, child: QueryExpr) -> Any:
        """Visit a child query and raise assertion errors if needed."""
        return child.accept(self)

    def visit_private_source(self, expr) -> Any:
        """Parse Turbo info from a PrivateSource query expression."""
        return None

    def visit_rename(self, expr: RenameExpr) -> Any:
        """Parse Turbo info from a Rename query expression."""
        raise ValueError("Turbo does not support Rename.")

    def visit_filter(self, expr: FilterExpr) -> Any:
        """Parse Turbo info from a FilterExpr query expression."""
        parsed = sqlglot.parse_one(expr.condition)
        for node in parsed.dfs():
            if not node[0].key in {
                "and",
                "or",
                "eq",
                "column",
                "identifier",
                "literal",
            }:
                raise ValueError("Turbo does not support binary-ops other than `=`")
        return expr.child.accept(self)

    def visit_select(self, expr: SelectExpr) -> Any:
        """Parse Turbo info from a Select query expression."""
        return expr.child.accept(self)

    def visit_map(self, expr: MapExpr) -> Any:
        """Parse Turbo info from a Map query expression."""
        raise ValueError("Turbo does not support Map.")

    def visit_flat_map(self, expr: FlatMapExpr) -> Any:
        """Parse Turbo info from a FlatMap query expression."""
        raise ValueError("Turbo does not support FlatMap.")

    def visit_join_private(self, expr: JoinPrivateExpr) -> Any:
        """Parse Turbo info from a JoinPrivate query expression."""
        raise ValueError("Turbo does not support JoinPrivate.")

    def visit_join_public(self, expr: JoinPublicExpr) -> Any:
        """Parse Turbo info from a JoinPublic query expression."""
        raise ValueError("Turbo does not support JoinPublic.")

    def visit_replace_null_and_nan(self, expr: ReplaceNullAndNan) -> Any:
        """Parse Turbo info from a ReplaceNullAndNan query expression."""
        raise ValueError("Turbo does not support ReplaceNullandNan.")

    def visit_replace_infinity(self, expr: ReplaceInfinity) -> Any:
        """Parse Turbo info from a ReplaceInfinity query expression."""
        raise ValueError("Turbo does not support ReplaceInfinity.")

    def visit_drop_infinity(self, expr: DropInfExpr) -> Any:
        """Parse Turbo info from a DropInfinity query expression."""
        raise ValueError("Turbo does not support DropInfinity.")

    def visit_drop_null_and_nan(self, expr: DropNullAndNan) -> Any:
        """Parse Turbo info from a DropNullAndNan query expression."""
        raise ValueError("Turbo does not support DropNullandNan.")

    def visit_enforce_constraint(self, expr: EnforceConstraint) -> Any:
        """Parse Turbo info from an EnforceConstraint query expression."""
        raise ValueError("Turbo does not support EnforceConstraint.")

    # None of the queries that produce measurements are implemented
    def visit_groupby_count(self, expr: GroupByCount) -> Any:
        """Visit a GroupByCount query expression."""
        if not expr.groupby_keys.dataframe().toPandas().empty:
            raise ValueError("Turbo does not support GroupBys.")
        return expr.child.accept(self)

    def visit_groupby_count_distinct(self, expr: GroupByCountDistinct) -> Any:
        """Visit a GroupByCountDistinct query expression (raises an error)."""
        raise ValueError("Turbo does not support count distinct")

    def visit_groupby_quantile(self, expr: GroupByQuantile) -> Any:
        """Visit a GroupByQuantile query expression (raises an error)."""
        raise ValueError("Turbo does not support quantile")

    def visit_groupby_bounded_sum(self, expr: GroupByBoundedSum) -> Any:
        """Visit a GroupByBoundedSum query expression (raises an error)."""
        raise ValueError("Turbo does not support bounded sum")

    def visit_groupby_bounded_average(self, expr: GroupByBoundedAverage) -> Any:
        """Visit a GroupByBoundedAverage query expression (raises an error)."""
        raise ValueError("Turbo does not support bounded average")

    def visit_groupby_bounded_variance(self, expr: GroupByBoundedVariance) -> Any:
        """Visit a GroupByBoundedVariance query expression (raises an error)."""
        raise ValueError("Turbo does not support bounded variance")

    def visit_groupby_bounded_stdev(self, expr: GroupByBoundedSTDEV) -> Any:
        """Visit a GroupByBoundedSTDEV query expression (raises an error)."""
        raise ValueError("Turbo does not support bounded stdev")


class DataViewIdVisitor(QueryVisitor):
    """A visitor to Parse Turbo info from a query expression."""

    def visit_private_source(self, expr) -> Any:
        """Parse Turbo info from a PrivateSource query expression."""
        return expr.source_id


class FilterClausesVisitor(QueryVisitor):
    """A visitor to Parse Turbo info from a query expression."""

    def visit_filter(self, expr: FilterExpr) -> Any:
        """Parse Turbo info from a FilterExpr query expression."""

        parsed = sqlglot.parse_one(expr.condition)

        filter_dict = {}
        eqs = list(parsed.find_all(sqlglot.exp.EQ))
        for eq in eqs:
            attr_name = eq.left.alias_or_name
            attr_value = eq.right.alias_or_name

            if attr_name not in filter_dict:
                filter_dict[attr_name] = [attr_value]
            else:
                filter_dict[attr_name] += [attr_value]
        return filter_dict


class AggregationTypeVisitor(QueryVisitor):
    """A visitor to Parse Turbo info from a query expression."""

    # None of the queries that produce measurements are implemented
    def visit_groupby_count(self, expr: GroupByCount) -> str:
        """Visit a GroupByCount query expression."""
        return "count"

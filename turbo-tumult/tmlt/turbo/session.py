from operator import xor
from typing import Any, Dict, Optional, Tuple, Union
from warnings import warn

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType
from tmlt.analytics._transformation_utils import get_table_from_ref
from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget
from tmlt.analytics.protected_change import AddMaxRows, ProtectedChange
from tmlt.analytics.query_expr import GroupByCount, QueryExpr
from tmlt.analytics.session import Session
from tmlt.core.measurements.interactive_measurements import (
    InactiveAccountantError,
    InsufficientBudgetError,
)
from tmlt.core.measures import PureDP
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.turbo import TumultTurboQuery, TumultDPEngineHook
from turbo.core import Accuracy, Turbo, calibrate_budget_pmwbypass
from typeguard import typechecked


def _format_insufficient_budget_msg(
    requested_budget: Union[ExactNumber, Tuple[ExactNumber, ExactNumber]],
    remaining_budget: Union[ExactNumber, Tuple[ExactNumber, ExactNumber]],
    privacy_budget: PrivacyBudget,
) -> str:
    """Format message for InsufficientBudgetError."""
    output = ""


class TurboSession(Session):
    """Extends Tumult analytic's Session class to evaluate queries running through Turbo."""

    def __init__(self, session):
        super().__init__(
            session._accountant, session._public_sources, session._compiler
        )

    # pylint: disable=line-too-long
    @classmethod
    @typechecked
    def from_dataframe(
        cls,
        privacy_budget: PrivacyBudget,
        source_id: str,
        dataframe: DataFrame,
        turbo_config: Dict[str, Any],
        stability: Optional[Union[int, float]] = None,
        grouping_column: Optional[str] = None,
        protected_change: Optional[ProtectedChange] = None,
    ) -> "TurboSession":
        # pylint: enable=line-too-long
        session_builder = (
            Session.Builder()
            .with_privacy_budget(privacy_budget=privacy_budget)
            .with_private_dataframe(
                source_id=source_id,
                dataframe=dataframe,
                stability=stability,
                grouping_column=grouping_column,
                protected_change=protected_change,
            )
        )
        if (
            not isinstance(protected_change, AddMaxRows)
            or protected_change.max_rows != 2
        ):
            raise ValueError(
                """Turbo works only with the ReplaceOneRow definition which Tumult doesn't support.
                             You must use AddMaxRows(2) which entails ReplaceOneRow instead"""
            )

        session = session_builder.build()

        # Use the created Session to create a Turbo session
        session = TurboSession(session)

        # Add Turbo in the session
        session.turbo = Turbo(
            config=turbo_config,
            dp_engine_hook=TumultDPEngineHook(session._accountant),
        )
        return session

    def evaluate(
        self, query_expr: QueryExpr, dp_demand: Union[PrivacyBudget, Accuracy]
    ) -> DataFrame:

        #################################################################################################
        # At this point we need the `privacy_budget`.
        # If an `accuracy` argument was passed instead of `privacy_budget` then we must convert it.
        # Assuming the `query_expr` always ends in an aggregation function we strip
        # the aggregation from the `query_expr` temporarily so that we can build a transformation from it.
        # From this transformation we can then compute the `data-view-size` and also obtain its `stability`.

        # Build the transformation
        transformation, reference = self._compiler.build_transformation(
            query=query_expr.child,
            input_domain=self._input_domain,
            input_metric=self._input_metric,
            public_sources=self._public_sources,
            catalog=self._catalog,
            table_constraints=self._table_constraints,
        )[:2]

        transformation = get_table_from_ref(transformation, reference)
        data_view_size = self.turbo.dp_engine_hook.get_data_view_size(
            TumultTurboQuery(None, transformation, None, None)
        )
        sensitivity_u2 = transformation.stability_function(
            self._accountant.d_in
        ).to_float(round_up=True)
        # `sensitivity_u2` = \Delta_{U2} must be 2 because Tumult is using the `AddMaxRows(2)` definition
        # and the query is a counting query.
        assert sensitivity_u2 == 2

        if isinstance(dp_demand, Accuracy):
            if not isinstance(query_expr, GroupByCount):
                raise ValueError(
                    "Can't request for accuracy target unless using Count."
                )

            if self._accountant.output_measure != PureDP():
                raise ValueError(
                    "Can't request for accuracy target unless using PureDP."
                )

            # Convert to `privacy_budget` thanks to Turbo's loose and simple upper bound
            # We could also optimize utility by using a lower budget for Bypass branches and pure Tumult queries
            # That requires two different measurements for the same query though
            # because for SV hard queries we need to fit both the SV false negative probability and the Laplace tail into beta

            # TODO: can we get rid of this call, and only calibrate once, for queries that are handled by Turbo?
            epsilon_u2 = calibrate_budget_pmwbypass(
                sensitivity_u2, dp_demand.alpha, dp_demand.beta, data_view_size
            )
            privacy_budget = PureDPBudget(epsilon_u2)
        else:
            privacy_budget = dp_demand
        #################################################################################################

        # pylint: enable=line-too-long
        measurement, adjusted_budget = self._compile_and_get_budget(
            query_expr, privacy_budget
        )
        self._activate_accountant()

        # check if type of self._accountant.privacy_budget matches adjusted_budget value
        if xor(
            isinstance(self._accountant.privacy_budget, tuple),
            isinstance(adjusted_budget.value, tuple),
        ):
            raise ValueError(
                "Expected type of adjusted_budget to match type of accountant's privacy"
                f" budget ({type(self._accountant.privacy_budget)}), but instead"
                f" received {type(adjusted_budget.value)}. This is probably a bug;"
                " please let us know about it so we can fix it!"
            )

        try:
            if not measurement.privacy_relation(
                self._accountant.d_in, adjusted_budget.value
            ):
                raise AssertionError(
                    "With these inputs and this privacy budget, similar inputs will"
                    " *not* produce similar outputs. This is probably a bug; please let"
                    " us know about it so we can fix it!"
                )

            try:
                # Try to execute measurement through Turbo
                if self._accountant.output_measure != PureDP():
                    raise ValueError("Turbo works only with PureDP.")

                # Compute `needed_privacy_budget` from Turbo's accuracy target
                turbo_alpha, turbo_beta = (
                    self.turbo.accuracy.alpha,
                    self.turbo.accuracy.beta,
                )

                # Do not use the conversion formula unless running a Count
                if not isinstance(query_expr, GroupByCount):
                    raise ValueError(
                        "Turbo does not support aggregations other than Counts"
                    )
                # For count queries, adding Lap(sensitivity_u2 / epsilon_u2) = Lap(sensitivity_b / epsilon_b) noise
                # satisfies epsilon_u2-U2DP but also epsilon_b-BDP.
                # We use turbo_alpha instead of dp_demand.alpha
                sensitivity_b = sensitivity_u2 / 2
                epsilon_b = calibrate_budget_pmwbypass(
                    sensitivity_b, turbo_alpha, turbo_beta, data_view_size
                )
                if epsilon_b != epsilon_u2 / 2:
                    raise ValueError(
                        "PrivacyBudget/Accuracy doesn't match Turbo's accuracy target."
                    )

                assert len(measurement.measurements) == 1
                measurement_ = measurement.measurements[0]

                # Normally the `privacy_budget` that we need to pay for a Laplace run could be inferred
                # by Turbo using the `calibrate_budget_pmwbypass` function. However, Turbo internally
                # works with a default sensitivity=1 (ReplaceOneRow).
                # Tumult does not support the `ReplaceOneRow` definition. Instead, we make it use the `AddMaxRows(2)` which entails the first (sensitivity=2).
                # This means that we will overpay when bypassing so that we are consistent with the definition.
                # Instead of confusing Turbo we pass the privacy budget that needs to be paid directly through the `TurboQuery`.
                turbo_query = TumultTurboQuery(
                    query_expr,
                    measurement_.transformation,
                    measurement_.measurement,  # Adds Lap(sensitivity_u2 / epsilon_u2) = Lap(sensitivity_b / epsilon_b) on Bypass queries
                    epsilon_b,  # We will pay only epsilon_b since we are epsilon_b-BDP
                )

                answer = self.turbo.run(turbo_query, self.turbo.accuracy)

                # Encapsulate the answer in a Spark Dataframe
                spark_schema = StructType([StructField("count", IntegerType())])
                answer = SparkSession.builder.getOrCreate().createDataFrame(
                    [[answer]], schema=spark_schema
                )
                return answer
            except Exception as exc:
                warn(
                    (f"Can't use Turbo, falling back to Tumult run. {exc}"),
                    RuntimeWarning,
                )

            # Try to execute measurement without Turbo
            # The measurement will use `dp_demand` = epsilon_u2 and satisfy epsilon_u2-U2DP
            # with Laplace noise or any other arbitrary Pure DP mechanism.
            # It will spend epsilon_u2.
            try:
                answers = self._accountant.measure(
                    measurement, d_out=adjusted_budget.value
                )
            except InsufficientBudgetError as err:
                msg = _format_insufficient_budget_msg(
                    err.requested_budget, err.remaining_budget, privacy_budget
                )
                raise RuntimeError(
                    "Cannot answer query without exceeding the Session privacy budget."
                    + msg
                ) from err

            if len(answers) != 1:
                raise AssertionError(
                    "Expected exactly one answer, but got "
                    f"{len(answers)} answers instead. This is "
                    "probably a bug; please let us know about it so "
                    "we can fix it!"
                )
            return answers[0]
        except InactiveAccountantError as e:
            raise RuntimeError(
                "This session is no longer active. Either it was manually stopped "
                "with session.stop(), or it was stopped indirectly by the "
                "activity of other sessions. See partition_and_create "
                "for more information."
            ) from e

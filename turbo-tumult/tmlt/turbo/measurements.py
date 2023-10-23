from tmlt.core.measurements.base import Measurement
from tmlt.core.utils.exact_number import ExactNumber, ExactNumberInput
from tmlt.core.measures import PureDP
from pyspark.sql import DataFrame
from typing import Any


class IdentityMeasurement(Measurement):
    """Directly forwards its input value without consuming any budget.
    Typically combined with an aggregation transformation to obtain the true aggregated result of a query
    without adding noise to it and without automatically calling the budget accountant.
    """

    def __init__(self, input_domain, input_metric):
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=PureDP(),
            is_interactive=False,
        )

    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        self.input_metric.validate(d_in)
        return 0

    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        return True

    def __call__(self, val: DataFrame) -> DataFrame:
        """Returns the actual output without adding any noise"""
        return val


class WrapperMeasurement(Measurement):
    """
    Wraps a measurement and calls it upon a `value`.
    Used when we want a measurement to add noise to an already available non DP output (true result)
    without having to re-compute it the entire pipeline of transformations.
    """

    def __init__(
        self,
        measurement,
        value,
    ):
        # Initialized from the measurement we wrap
        super().__init__(
            input_domain=measurement.input_domain,
            input_metric=measurement.input_metric,
            output_measure=PureDP(),
            is_interactive=False,
        )
        self.measurement = measurement
        self.value = value

    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        return self.measurement.privacy_function(d_in)

    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        return self.measurement.privacy_relation(d_in, d_out)

    def __call__(self, val: DataFrame) -> DataFrame:
        """Uses the previous DP measurement (e.g. AddLaplaceNoise) to add noise to an already available value"""
        noise_addition = self.measurement.measurement.measurement
        return noise_addition(self.value)


class BudgetConsumptionMeasurement(Measurement):
    """A measurement that just consumes budget and doesn't perform any operation"""

    def __init__(self, input_domain, input_metric, output_measure, d_out):
        super().__init__(
            input_domain=input_domain,
            input_metric=input_metric,
            output_measure=output_measure,
            is_interactive=False,
        )
        self.d_out = d_out

    def privacy_function(self, d_in: ExactNumberInput) -> ExactNumber:
        return self.d_out

    def privacy_relation(self, d_in: Any, d_out: Any) -> bool:
        return True

    def __call__(self, val: DataFrame) -> DataFrame:
        return None

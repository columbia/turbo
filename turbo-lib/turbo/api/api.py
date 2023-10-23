from abc import ABC, abstractmethod
from typing import Any, Optional

# Classes to be implemented by users fot integration with Turbo


class TurboQuery(ABC):
    """A class that represents a query. It has the ability to extract info about itself that is needed by Turbo.
    Users should extend this class to implement methods tailored for their query format.

    get_aggregation_type: Function that yields the aggregation type of the query. Must return the string "count" (no support for other aggregates for now).
    get_data_view_id: Function that yiels an id that uniquely identifies the data view on which the query will operate.
    get_filter_clause: Function that yields a dictionary with all the attributes that are assigned to specific values in the filtering condition.
    """

    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def get_filter_clause(self):
        raise NotImplementedError()

    @abstractmethod
    def get_aggregation_type(self):
        raise NotImplementedError()

    @abstractmethod
    def get_data_view_id(self):
        raise NotImplementedError()


class DPEngineHook(ABC):
    """A class with the ability to execute queries both with and without DP and consume budget
    Users should extend this class to implement methods tailored for their underlying DP engines.

    executeNPQuery: A function that helps Turbo retrieve the true output of the query
    executeDPQuery: A function that helps Turbo retrieve the DP output of the query
    consume: A function that consumes privacy budget
    get_data_view_size: Function that yields the size of the dataset-view on which the query will operate.
    (In Turbo, the data-view size is considered to be public knowledge. Refer to the paper for details)
    """

    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def executeNPQuery(self, query: TurboQuery):
        """Executes the query and returns the true result
        Args:
            query: A TurboQuery as specified by the users
        """
        raise NotImplementedError()

    @abstractmethod
    def executeDPQuery(
        self, query: TurboQuery, budget: float, true_result: Optional[Any]
    ):
        """Adds noise to a true result to produce a DP output and consumes privacy budget.
        Args:
            query: A TurboQuery as specified by the users
            budget: Budget to be spent for the computation (assuming stability=1)
            true_result: True result of the query

            Returns the noisy result and the budget that was really consumed
        """
        raise NotImplementedError()

    @abstractmethod
    def consume_budget(self, budget: float):
        """Consumes privacy budget.
        Args:
            budget: The budget that will be consumed.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_data_view_size(self, query: TurboQuery):
        """Calculates the size of the data view on which the query will operate.
        Args:
            query: A TurboQuery as specified by the users
        """
        raise NotImplementedError()

from abc import ABC, abstractmethod
import math
import pandas as pd


class BaseFairnessMetric(ABC):
    """
    Abstract base class defining the basic structure of a fairness metric.
    """

    @abstractmethod
    def compute(self,
                input_data: pd.DataFrame,
                sensitive_attrs: dict = None,
                input_features: list = None,
                column_output: str = None,
                positive_output: list = [1]):
        """
        Main method to compute the metric.
        Parameters
        ----------
        :param input_data: List or array of ground-truth (true) labels.
        :param sensitive_attrs: List or array of model predictions or scores.
        :param input_features: List of features use as input to get the output.
        :param column_output: Name of the column containing the target.
        :param positive_output: Values of the column_output consider as positive.
        Returns
        -------
        :return: A numeric value or a dictionary containing the metric results.
        """
        raise NotImplementedError("Not implemented.")

    def ratio_component(self, underprivileged, privileged, ratio_key):
        """
        Result block for a metric reported as the ratio of a rate between groups
        (underprivileged / privileged).

        A rate is undefined (NaN) when the group has no cases to compute it on, for
        example no positive predictions. The component is then reported as not
        computable, with empty values, instead of as a score of 0.

        :param underprivileged: Rate for the underprivileged group, NaN if undefined.
        :param privileged: Rate for the privileged group, NaN if undefined.
        :param ratio_key: Name of the key that holds the ratio.

        :return: A dictionary with the rates, their ratio, the normalized value and the bias level.
        """
        underprivileged = float(underprivileged)
        privileged = float(privileged)
        if math.isnan(underprivileged) or math.isnan(privileged):
            return {
                'underprivileged': None if math.isnan(underprivileged) else round(underprivileged, 4),
                'privileged': None if math.isnan(privileged) else round(privileged, 4),
                ratio_key: None,
                'normalized_risk': None,
                'bias_level': 'Not computable',
                'note': 'not computable: a group has no cases for this rate.'
            }
        if privileged == 0 and underprivileged == 0:
            # Equal rates: parity, although the ratio 0 / 0 is undefined.
            ratio = 1.0
        elif privileged == 0:
            # The ratio is unbounded: the lowest score, and no ratio to report.
            return {
                'underprivileged': round(underprivileged, 4),
                'privileged': 0.0,
                ratio_key: None,
                'normalized_risk': 0.0,
                'bias_level': self.get_bias_level(0),
                'note': 'ratio unbounded: the privileged rate is 0.'
            }
        else:
            ratio = round(underprivileged / privileged, 4)
        normalized = self.normalize_value(ratio)
        return {
            'underprivileged': round(underprivileged, 4),
            'privileged': round(privileged, 4),
            ratio_key: ratio,
            'normalized_risk': round(normalized, 4),
            'bias_level': self.get_bias_level(normalized)
        }

    def normalize_value(self, value):
        """
        Main method to normalize the metric between 0 (BAD) to 100 (GOOD).

        :param value: Metric value.

        :return: A numeric value between 0 to 100.
        """
        raise NotImplementedError("Not implemented.")

    def get_bias_level(self, value):
        """
        Main method to get the bias level.

        :param value: Metric value.

        :return:
        """
        return 'Low/no bias' if value > 80 else 'High bias' if value <= 60 else 'Medium bias'

    def validate_parameters(self, **kwargs):
        """
        check input params are null or not.
        """
        for param_name, param_value in kwargs.items():
            if param_value is None:
                raise ValueError(f"You must provide a value for {param_name} to compute {self.__class__.__name__}")

    def __str__(self):
        """
        String representation of the metric.
        """
        return self.__class__.__name__

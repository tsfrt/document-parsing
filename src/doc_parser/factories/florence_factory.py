"""Factory script for the Florence-2 pyfunc model. Used by `mlflow.pyfunc.log_model`."""

import mlflow.models

from doc_parser.models.florence import FlorencePyfunc

mlflow.models.set_model(FlorencePyfunc())

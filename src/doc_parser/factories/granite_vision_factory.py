"""Factory script for the Granite-Vision pyfunc model. Used by `mlflow.pyfunc.log_model`."""

import mlflow.models

from doc_parser.models.granite_vision import GraniteVisionPyfunc

mlflow.models.set_model(GraniteVisionPyfunc())

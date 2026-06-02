"""Factory script for the Phi-3.5-vision pyfunc model. Used by `mlflow.pyfunc.log_model`."""

import mlflow.models

from doc_parser.models.phi3_vision import Phi3VisionPyfunc

mlflow.models.set_model(Phi3VisionPyfunc())

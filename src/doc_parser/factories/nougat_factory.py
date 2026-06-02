"""Factory script for the Nougat pyfunc model. Used by `mlflow.pyfunc.log_model`."""

import mlflow.models

from doc_parser.models.nougat import NougatPyfunc

mlflow.models.set_model(NougatPyfunc())

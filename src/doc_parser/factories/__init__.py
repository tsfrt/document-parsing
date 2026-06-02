"""Factory entry-points used by ``mlflow.pyfunc.log_model(python_model=<path>)``.

Each factory module imports its concrete OCR pyfunc class and calls
``mlflow.models.set_model(...)`` so MLflow can serialize the model definition
as source code rather than pickling a class instance. This sidesteps a
cloudpickle re-hydration bug we saw on Databricks Model Serving where the
pickled class lookup couldn't resolve the wrapper class even though the
``code_paths`` directory was on sys.path.
"""

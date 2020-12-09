import os

from bentoml.exceptions import InvalidArgument, MissingDependencyException
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.service.env import BentoServiceEnv


class CatboostModelArtifact(BentoServiceArtifact):
    """Abstraction for save/load object with Xgboost.

    Args:
        name (string): name of the artifact
        model_extension (string): Extension name for saved xgboost model

    Raises:
        ImportError: catboost package is required for using CatboostModelArtifact
        TypeError: invalid argument type, model being packed must be instance of
            catboost.core.CatBoostClassifier

    Example usage:

    >>> from catboost import CatBoostClassifier
    >>>
    >>> import bentoml
    >>> from bentoml.frameworks.catboost import CatboostModelArtifact
    >>> from bentoml.adapters import DataframeInput
    >>>
    >>> @bentoml.env(infer_pip_packages=True)
    >>> @bentoml.artifacts(CatboostModelArtifact('model'))
    >>> class CatBoostModelService(bentoml.BentoService):
    >>>
    >>>     @bentoml.api(input=DataframeInput(), batch=True)
    >>>     def predict(self, df):
    >>>         result = self.artifacts.model.predict(df)
    >>>         return result
    >>>
    >>> svc = CatBoostModelService()
    >>> # Pack catboost model
    >>> svc.pack('model', model_to_save)
    """

    def __init__(self, name, model_extension=".json"):
        super(CatboostModelArtifact, self).__init__(name)
        self._model_extension = model_extension
        self._model = None

    def set_dependencies(self, env: BentoServiceEnv):
        env.add_pip_packages(['catboost'])

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._model_extension)

    def pack(self, model, metadata=None):  # pylint:disable=arguments-differ
        try:
            import catboost
            from catboost import CatBoostClassifier
        except ImportError:
            raise MissingDependencyException(
                "catboost package is required to use CatboostModelArtifact"
            )

        if not isinstance(model, catboost.core.CatBoostClassifier):
            raise InvalidArgument(
                "Expect `model` argument to be a `catboost.core.CatBoostClassifier` instance"
            )

        self._model = model
        return self

    def load(self, path):
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            raise MissingDependencyException(
                "catboost package is required to use CatboostModelArtifact"
            )
        clf = CatBoostClassifier()
        clf.load_model(self._model_file_path(path), "json")
        return self.pack(clf)

    def save(self, dst):
        return self._model.save_model(self._model_file_path(dst), format="json")

    def get(self):
        return self._model

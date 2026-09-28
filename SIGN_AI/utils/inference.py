"""Read the stored scaler and class labels without importing scikit-learn."""

import pickle
import numpy as np


class InferenceScaler:
    def transform(self, frames):
        return ((frames - self.mean_) / self.scale_).astype(np.float32, copy=False)


class InferenceLabels:
    pass


class _InferenceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("sklearn.preprocessing._data", "StandardScaler"):
            return InferenceScaler
        if (module, name) == ("sklearn.preprocessing._label", "LabelEncoder"):
            return InferenceLabels
        return super().find_class(module, name)


def load_inference_pickle(path):
    with open(path, "rb") as artifact:
        return _InferenceUnpickler(artifact).load()

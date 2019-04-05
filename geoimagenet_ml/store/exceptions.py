#!/usr/bin/env python
# coding: utf-8


class InvalidIdentifierValue(ValueError):
    """
    Error indicating that an id to be employed for following operations
    is not considered as valid to allow further processed or usage.
    """
    pass


class DatasetNotFoundError(Exception):
    """
    Error indicating that a dataset could not be read from the
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.DatasetStore`.
    """
    pass


class DatasetRegistrationError(Exception):
    """
    Error indicating that a dataset could not be registered in the
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.DatasetStore`.
    """
    pass


class DatasetInstanceError(Exception):
    """
    Error indicating that the instance passed is not supported with
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.DatasetStore`.
    """
    pass


class DatasetConflictError(Exception):
    """
    Error indicating that the instance of :class:`geoimagenet_ml.api.store.interfaces.DatasetStore`
    is conflicting with another process in the storage backend .
    """
    pass


class ModelNotFoundError(Exception):
    """
    Error indicating that a model could not be read from the
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.ModelStore`.
    """
    pass


class ModelRegistrationError(Exception):
    """
    Error indicating that a model could not be registered in the
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.ModelStore`.
    """
    pass


class ModelInstanceError(Exception):
    """
    Error indicating that the instance passed is not supported with
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.ModelStore`.
    """
    pass


class ModelLoadingError(Exception):
    """
    Error indicating that loading of the model data from the model definition failed.
    """
    pass


class ModelConflictError(Exception):
    """
    Error indicating that the instance of :class:`geoimagenet_ml.api.store.interfaces.ModelStore`
    is conflicting with another process in the storage backend .
    """
    pass


class ProcessNotFoundError(Exception):
    """
    Error indicating that a process could not be read from the
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.ProcessStore`.
    """
    pass


class ProcessRegistrationError(Exception):
    """
    Error indicating that a process could not be registered in the
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.ProcessStore`.
    """
    pass


class ProcessInstanceError(Exception):
    """
    Error indicating that the instance passed is not supported with
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.ProcessStore`.
    """
    pass


class ProcessConflictError(Exception):
    """
    Error indicating that the instance of :class:`geoimagenet_ml.api.store.interfaces.ProcessStore`
    is conflicting with another process in the storage backend .
    """
    pass


class JobNotFoundError(Exception):
    """
    Error indicating that a job could not be read from the
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.JobStore`.
    """
    pass


class JobRegistrationError(Exception):
    """
    Error indicating that a job could not be registered in the
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.JobStore`.
    """
    pass


class JobUpdateError(Exception):
    """
    Error indicating that a job could not be updated in the
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.JobStore`.
    """
    pass


class JobInstanceError(Exception):
    """
    Error indicating that the instance passed is not supported with
    storage backend by an instance of :class:`geoimagenet_ml.api.store.interfaces.JobStore`.
    """
    pass


class JobConflictError(Exception):
    """
    Error indicating that the instance of :class:`geoimagenet_ml.api.store.interfaces.JobStore`
    is conflicting with another process in the storage backend .
    """
    pass

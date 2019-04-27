#!/usr/bin/env python
# coding: utf-8


class InvalidIdentifierValue(ValueError):
    """
    Error indicating that an id to be employed for following operations
    is not considered as valid to allow further processed or usage.
    """
    pass


class DatasetError(Exception):
    """
    Error related to :class:`geoimagenet_ml.store.datatypes.Dataset`.
    """
    pass


class DatasetNotFoundError(DatasetError):
    """
    Error indicating that a dataset could not be read from the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.DatasetStore`.
    """
    pass


class DatasetRegistrationError(DatasetError):
    """
    Error indicating that a dataset could not be registered in the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.DatasetStore`.
    """
    pass


class DatasetInstanceError(DatasetError):
    """
    Error indicating that the instance passed is not supported with
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.DatasetStore`.
    """
    pass


class DatasetConflictError(DatasetError):
    """
    Error indicating that the instance of :class:`geoimagenet_ml.store.interfaces.DatasetStore`
    is conflicting with another process in the storage backend .
    """
    pass


class ModelError(Exception):
    """
    Error related to :class:`geoimagenet_ml.store.datatypes.Model`.
    """
    pass


class ModelNotFoundError(ModelError):
    """
    Error indicating that a model could not be read from the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.ModelStore`.
    """
    pass


class ModelRegistrationError(ModelError):
    """
    Error indicating that a model could not be registered in the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.ModelStore`.
    """
    pass


class ModelInstanceError(ModelError):
    """
    Error indicating that the instance passed is not supported with
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.ModelStore`.
    """
    pass


class ModelLoadingError(ModelError):
    """
    Error indicating that loading of the model data from the model definition failed.
    """
    pass


class ModelConflictError(ModelError):
    """
    Error indicating that the instance of :class:`geoimagenet_ml.store.interfaces.ModelStore`
    is conflicting with another process in the storage backend .
    """
    pass


class ProcessError(Exception):
    """
    Error related to :class:`geoimagenet_ml.store.datatypes.Process`.
    """
    pass


class ProcessNotFoundError(ProcessError):
    """
    Error indicating that a process could not be read from the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.ProcessStore`.
    """
    pass


class ProcessRegistrationError(ProcessError):
    """
    Error indicating that a process could not be registered in the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.ProcessStore`.
    """
    pass


class ProcessInstanceError(ProcessError):
    """
    Error indicating that the instance passed is not supported with
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.ProcessStore`.
    """
    pass


class ProcessConflictError(ProcessError):
    """
    Error indicating that the instance of :class:`geoimagenet_ml.store.interfaces.ProcessStore`
    is conflicting with another process in the storage backend .
    """
    pass


class JobError(Exception):
    """
    Error related to :class:`geoimagenet_ml.store.datatypes.Job`.
    """
    pass


class JobNotFoundError(JobError):
    """
    Error indicating that a job could not be read from the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.JobStore`.
    """
    pass


class JobRegistrationError(JobError):
    """
    Error indicating that a job could not be registered in the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.JobStore`.
    """
    pass


class JobUpdateError(JobError):
    """
    Error indicating that a job could not be updated in the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.JobStore`.
    """
    pass


class JobInstanceError(JobError):
    """
    Error indicating that the instance passed is not supported with
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.JobStore`.
    """
    pass


class JobConflictError(JobError):
    """
    Error indicating that the instance of :class:`geoimagenet_ml.store.interfaces.JobStore`
    is conflicting with another process in the storage backend .
    """
    pass


class ActionError(Exception):
    """
    Error related to :class:`geoimagenet_ml.store.datatypes.Action`.
    """
    pass


class ActionInstanceError(ActionError):
    """
    Error indicating that instance of :class:`geoimagenet_ml.store.datatypes.Action`
    could not be successfully generated using provided arguments.
    """
    pass


class ActionRegistrationError(ActionError):
    """
    Error indicating that a job could not be registered in the
    storage backend by an instance of :class:`geoimagenet_ml.store.interfaces.ActionStore`.
    """
    pass

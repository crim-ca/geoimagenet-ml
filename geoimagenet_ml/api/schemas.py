#!/usr/bin/env python
# coding: utf-8

from geoimagenet_ml import __meta__
from geoimagenet_ml.status import STATUS
from geoimagenet_ml.constants import VISIBILITY
from cornice.service import Service
from colander import drop, Boolean, DateTime, Integer, MappingSchema, OneOf, SchemaNode, SequenceSchema, String, Time
from pyramid.httpexceptions import (
    HTTPOk,
    HTTPCreated,
    HTTPAccepted,
    HTTPBadRequest,
    HTTPUnauthorized,
    HTTPForbidden,
    HTTPNotFound,
    HTTPConflict,
    HTTPUnprocessableEntity,
    HTTPInternalServerError,
)
import six


TitleAPI = "GeoImageNet ML REST API"
InfoAPI = {
    "description": __meta__.__description__,
    "contact": {"name": __meta__.__author__, "email": __meta__.__email__, "url": __meta__.__url__}
}


class CorniceSwaggerPredicate(object):
    """Predicate to add simple information to Cornice Swagger."""

    # noinspection PyUnusedLocal
    def __init__(self, schema, config):
        self.schema = schema

    def phash(self):
        return str(self.schema)

    def __call__(self, context, request):
        return self.schema


URL = "url"

# Tags
TagAPI = "API"
TagDatasets = "Datasets"
TagModels = "Models"
TagProcesses = "Processes"
TagJobs = "Jobs"

# Route parameters
ParamDatasetUUID = "dataset_uuid"
ParamProcessUUID = "process_uuid"
ParamModelUUID = "model_uuid"
ParamJobUUID = "job_uuid"


def make_param(variable):
    return "{" + variable + "}"


# Route parameters literals
VariableDatasetUUID = make_param(ParamDatasetUUID)
VariableProcessUUID = make_param(ParamProcessUUID)
VariableModelUUID = make_param(ParamModelUUID)
VariableJobUUID = make_param(ParamJobUUID)


# Service Routes
BaseAPI = Service(
    path="/",
    name=__meta__.__title__,
    description="GeoImageNet ML REST API information. Base path of the API.")
SwaggerJSON = Service(
    path=BaseAPI.path + "json",
    name=__meta__.__title__ + "swagger_schemas",
    description="Schemas of {}".format(__meta__.__title__))
SwaggerAPI = Service(
    path=BaseAPI.path + "api",
    name="swagger",
    description="Swagger of {}".format(__meta__.__title__))
DatasetsAPI = Service(
    path=BaseAPI.path + "datasets",
    name="Datasets")
DatasetLatestAPI = Service(
    path=BaseAPI.path + "datasets/latest",
    name="DatasetLatest")
DatasetAPI = Service(
    path=BaseAPI.path + "datasets/" + VariableDatasetUUID,
    name="Dataset")
DatasetDownloadAPI = Service(
    path=BaseAPI.path + "datasets/" + VariableDatasetUUID + "/download",
    name="DatasetDownload")
ModelsAPI = Service(
    path=BaseAPI.path + "models",
    name="Models")
ModelAPI = Service(
    path=BaseAPI.path + "models/" + VariableModelUUID,
    name="Model")
ModelDownloadAPI = Service(
    path=BaseAPI.path + "models/" + VariableModelUUID + "/download",
    name="ModelDownload")
ModelStatisticsAPI = Service(
    path=BaseAPI.path + "models/" + VariableModelUUID + "/statistics",
    name="ModelStatistics")
ProcessesAPI = Service(
    path=BaseAPI.path + "processes",
    name="Processes")
ProcessAPI = Service(
    path=BaseAPI.path + "processes/" + VariableProcessUUID,
    name="Process")
ProcessJobsAPI = Service(
    path=BaseAPI.path + "processes/" + VariableProcessUUID + "/jobs",
    name="ProcessJobs")
ProcessJobCurrentAPI = Service(
    path=BaseAPI.path + "processes/" + VariableProcessUUID + "/jobs/current",
    name="ProcessJobCurrent")
ProcessJobLatestAPI = Service(
    path=BaseAPI.path + "processes/" + VariableProcessUUID + "/jobs/latest",
    name="ProcessJobLatest")
ProcessJobAPI = Service(
    path=BaseAPI.path + "processes/" + VariableProcessUUID + "/jobs/" + VariableJobUUID,
    name="ProcessJob")
ProcessJobResultAPI = Service(
    path=BaseAPI.path + "processes/" + VariableProcessUUID + "/jobs/" + VariableJobUUID + "/result",
    name="ProcessJobResult")
ProcessJobLogsAPI = Service(
    path=BaseAPI.path + "processes/" + VariableProcessUUID + "/jobs/" + VariableJobUUID + "/logs",
    name="ProcessJobLogs")
ProcessJobExceptionsAPI = Service(
    path=BaseAPI.path + "processes/" + VariableProcessUUID + "/jobs/" + VariableJobUUID + "/exceptions",
    name="ProcessJobExceptions")
VersionsAPI = Service(
    path=BaseAPI.path + "versions",
    name="Versions")

# Generic Endpoint parts
dataset_uuid = SchemaNode(String(), description="Dataset UUID.", title="Dataset UUID.")
model_uuid = SchemaNode(String(), description="Model UUID.", title="Model UUID.")
process_uuid = SchemaNode(String(), description="Process UUID.", title="Process UUID or Identifier.")
job_uuid = SchemaNode(String(), description="Job UUID.", title="Job UUID.")

# Security
SecurityDefinitionAPI = {"securityDefinitions": {"cookieAuth": {"type": "apiKey", "in": "cookie", "name": "auth_tkt"}}}
SecurityAdministratorAPI = [{"cookieAuth": []}]
SecurityEveryoneAPI = []

# Content
ContentTypeJSON = "application/json"
ContentTypeHTML = "text/html"


# Service Routes Utility
def service_api_route_info(service_api):
    return {"name": service_api.name, "pattern": service_api.path}


def get_security(service, method):
    definitions = service.definitions
    args = {}
    for definition in definitions:
        met, view, args = definition
        if met == method:
            break
    return SecurityAdministratorAPI if "security" not in args else args["security"]


class HeaderSchemaJSON(MappingSchema):
    content_type = SchemaNode(String(), example=ContentTypeJSON, default=ContentTypeJSON)
    content_type.name = "Content-Type"


class HeaderSchemaHTML(MappingSchema):
    content_type = SchemaNode(String(), example=ContentTypeHTML, default=ContentTypeHTML)
    content_type.name = "Content-Type"


class AcceptHeader(MappingSchema):
    Accept = SchemaNode(String(), missing=drop, default=ContentTypeJSON,
                        validator=OneOf([ContentTypeJSON, ContentTypeHTML]))


class BaseRequestSchema(MappingSchema):
    header = AcceptHeader()


class BaseMetaResponseSchema(MappingSchema):
    code = SchemaNode(
        Integer(), description="HTTP response code.", example=HTTPOk.code)
    type = SchemaNode(
        String(), description="Response content type.", example="application/json")
    detail = SchemaNode(
        String(), description="Response status message.")
    route = SchemaNode(
        String(), description="Request route called that generated the response.", missing=drop)
    uri = SchemaNode(
        String(), description="Request URI that generated the response.", missing=drop)
    method = SchemaNode(
        String(), description="Request method that generated the response.", missing=drop)


class BaseBodyResponseSchema(MappingSchema):
    meta = BaseMetaResponseSchema()
    data = MappingSchema(default={})

    __code = None
    __desc = None

    def __init__(self, code, description):
        super(BaseBodyResponseSchema, self).__init__()
        assert isinstance(code, int)
        assert isinstance(description, six.string_types)
        self.__code = code
        self.__desc = description

        # update the values
        child_nodes = getattr(self, "children")
        for node in child_nodes:
            if node.name == "meta":
                for meta_node in getattr(node, "children"):
                    if meta_node.name == "code":
                        meta_node.example = self.__code
                    if meta_node.name == "detail":
                        meta_node.example = self.__desc


class BaseResponseSchema(MappingSchema):
    description = "UNDEFINED"
    header = AcceptHeader()
    body = BaseBodyResponseSchema(code=HTTPOk.code, description=description)


class ErrorBodyResponseSchema(BaseBodyResponseSchema):
    data = MappingSchema()


class UnauthorizedDataResponseSchema(MappingSchema):
    route_name = SchemaNode(String(), description="Specified route")
    request_url = SchemaNode(String(), description="Specified url")


class UnauthorizedResponseSchema(BaseResponseSchema):
    description = "Unauthorized. Insufficient user privileges or missing authentication headers."
    body = ErrorBodyResponseSchema(code=HTTPUnauthorized.code, description=description)


class NotFoundResponseSchema(BaseResponseSchema):
    description = "The route resource could not be found."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class UnprocessableEntityResponseSchema(BaseResponseSchema):
    description = "Invalid value specified."
    body = ErrorBodyResponseSchema(code=HTTPUnprocessableEntity.code, description=description)


class TracebackListSchema(SequenceSchema):
    traceback_item = SchemaNode(String(), missing=drop, description="Summary line of the traceback.")


class InternalServerErrorDataSchema(MappingSchema):
    exception = SchemaNode(String(), missing=drop, description="Exception message description.")
    traceback = TracebackListSchema(default=[], missing=drop,
                                    description="Exception stack trace caused by the request.")
    caller = MappingSchema(default={}, missing=drop,
                           description="Details of the calling request generating this error.")


class InternalServerErrorBodySchema(ErrorBodyResponseSchema):
    def __init__(self, description):
        super(InternalServerErrorBodySchema, self).\
            __init__(code=HTTPInternalServerError.code, description=description)

    error = InternalServerErrorDataSchema(description="Details of the generated error.")


class InternalServerErrorResponseSchema(BaseResponseSchema):
    description = "Internal Server Error. Unhandled exception occurred."
    body = InternalServerErrorBodySchema(description=description)


class FileList(SequenceSchema):
    file = SchemaNode(String(), description="File path.")


class DatasetSummaryBodyResponseSchema(MappingSchema):
    uuid = SchemaNode(String(), description="Dataset uuid.", title="UUID")
    name = SchemaNode(String(), description="Dataset name.")


class DatasetDetailBodyResponseSchema(DatasetSummaryBodyResponseSchema):
    type = SchemaNode(String(), description="Dataset type.")
    path = SchemaNode(String(), description="Dataset path.")
    data = MappingSchema(description="Dataset complete data definition (no specific format, depends on dataset type).")
    files = FileList(description="Files referenced internally by the dataset.")
    status = SchemaNode(String(), description="Dataset status.", example=STATUS.FINISHED.value)
    created = SchemaNode(DateTime(), description="Dataset creation time (not complete).")
    finished = SchemaNode(DateTime(), description="Dataset completion time.")


class DatasetSummaryListSchema(SequenceSchema):
    dataset_summary = DatasetSummaryBodyResponseSchema()


class Datasets_GET_DataResponseSchema(MappingSchema):
    datasets = DatasetSummaryListSchema()


class Datasets_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Datasets_GET_DataResponseSchema()


class Datasets_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get datasets successful."
    body = Datasets_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class Datasets_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Get datasets by name query refused by db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class Datasets_POST_BodyRequestSchema(MappingSchema):
    dataset_name = SchemaNode(String(), description="Name of the new dataset.")
    dataset_path = SchemaNode(String(), description="Path of the new dataset.")


class Datasets_POST_RequestSchema(MappingSchema):
    header = HeaderSchemaJSON()
    body = Datasets_POST_BodyRequestSchema()


class Dataset_POST_DataResponseSchema(MappingSchema):
    dataset = DatasetDetailBodyResponseSchema()


class Datasets_POST_BodyResponseSchema(BaseBodyResponseSchema):
    data = Dataset_POST_DataResponseSchema()


class Datasets_POST_CreatedResponseSchema(BaseResponseSchema):
    description = "Create dataset successful."
    body = Datasets_POST_BodyResponseSchema(code=HTTPCreated.code, description=description)


class Datasets_POST_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to create dataset."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class Datasets_POST_NotFoundResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve created dataset from db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class Datasets_POST_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to add dataset to db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class Datasets_POST_ConflictResponseSchema(BaseResponseSchema):
    description = "Dataset conflicts with an already existing dataset."
    body = ErrorBodyResponseSchema(code=HTTPConflict.code, description=description)


class DatasetEndpoint(BaseRequestSchema):
    dataset_uuid = dataset_uuid


class Dataset_GET_DataResponseSchema(MappingSchema):
    dataset = DatasetDetailBodyResponseSchema()
    owner = SchemaNode(Integer(), description="User ID of the dataset owner (uploader or creator).", default=None)
    downloads = SchemaNode(Integer(), description="Number of time this dataset was downloaded.", default=0)


class Dataset_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Dataset_GET_DataResponseSchema()


class Dataset_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get dataset successful."
    body = Dataset_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class Dataset_GET_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to retrieve dataset."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class Dataset_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve dataset from db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class Dataset_GET_NotFoundResponseSchema(BaseResponseSchema):
    description = "Dataset could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class Dataset_DELETE_BodyResponseSchema(BaseBodyResponseSchema):
    data = MappingSchema()


class Dataset_DELETE_OkResponseSchema(BaseResponseSchema):
    description = "Delete dataset successful."
    body = Dataset_DELETE_BodyResponseSchema(code=HTTPOk.code, description=description)


class Dataset_DELETE_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to retrieve dataset."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class Dataset_DELETE_NotFoundResponseSchema(BaseResponseSchema):
    description = "Dataset could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class DatasetLatest_GET_BodyRequestSchema(MappingSchema):
    dataset_name = SchemaNode(String(), missing=drop,
                              description="Filter search only to datasets matching the specified name.")
    dataset_type = SchemaNode(String(), missing=drop,
                              description="Filter search only to datasets matching the specified type.")


class DatasetLatestEndpoint(BaseRequestSchema):
    body = DatasetLatest_GET_BodyRequestSchema()


class DatasetDownloadEndpoint(BaseRequestSchema):
    dataset_uuid = dataset_uuid


class DatasetDownload_GET_OkResponseSchema(BaseResponseSchema):
    description = "Dataset download successful."
    body = BaseBodyResponseSchema(code=HTTPOk.code, description=description)


class DatasetDownload_GET_NotFoundResponseSchema(BaseResponseSchema):
    description = "Dataset download file could not be found."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class ModelSummaryBodyResponseSchema(MappingSchema):
    uuid = SchemaNode(String(), description="Model uuid.", title="UUID")
    name = SchemaNode(String(), description="Model name.")


class ModelDetailBodyResponseSchema(ModelSummaryBodyResponseSchema):
    path = SchemaNode(String(), description="Model path.")
    created = SchemaNode(DateTime(), description="Model creation time.")


class ModelSummaryListSchema(SequenceSchema):
    model_summary = ModelSummaryBodyResponseSchema()


class Models_GET_DataResponseSchema(MappingSchema):
    models = ModelSummaryListSchema()


class Models_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Models_GET_DataResponseSchema()


class Models_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get models successful."
    body = Models_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class Models_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Get models by name query refused by db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class Model_GET_DataResponseSchema(MappingSchema):
    model = ModelDetailBodyResponseSchema(description="Detailed information of the model.")


class Model_POST_DataResponseSchema(MappingSchema):
    model = ModelDetailBodyResponseSchema(description="Detailed information of the model.")
    owner = SchemaNode(Integer(), description="User ID of the model owner (uploader or creator).", default=None)
    downloads = SchemaNode(Integer(), description="Number of time this model was downloaded.", default=0)


class Models_POST_BodyRequestSchema(MappingSchema):
    model_name = SchemaNode(String(), description="Name of the new model.")
    model_path = SchemaNode(String(), description="Path of the new model.")


class Models_POST_RequestSchema(MappingSchema):
    header = HeaderSchemaJSON()
    body = Models_POST_BodyRequestSchema()


class Models_POST_BodyResponseSchema(BaseBodyResponseSchema):
    data = Model_POST_DataResponseSchema()


class Models_POST_CreatedResponseSchema(BaseResponseSchema):
    description = "Create model successful."
    body = Models_POST_BodyResponseSchema(code=HTTPCreated.code, description=description)


class Models_POST_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to create model."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class Models_POST_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to add model to db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class Models_POST_NotFoundResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve created model from db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class Models_POST_ConflictResponseSchema(BaseResponseSchema):
    description = "Model conflicts with an already existing model."
    body = ErrorBodyResponseSchema(code=HTTPConflict.code, description=description)


class Models_POST_UnprocessableEntityResponseSchema(BaseResponseSchema):
    description = "Model cannot be loaded from file."
    body = ErrorBodyResponseSchema(code=HTTPUnprocessableEntity.code, description=description)


class Model_GET_Endpoint(BaseRequestSchema):
    model_uuid = model_uuid


class Model_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Model_GET_DataResponseSchema()


class Model_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get model successful."
    body = Model_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class Model_GET_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to retrieve model."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class Model_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve model from db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class Model_GET_NotFoundResponseSchema(BaseResponseSchema):
    description = "Model could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class ModelDownloadEndpoint(BaseRequestSchema):
    model_uuid = model_uuid


class ModelDownload_GET_OkResponseSchema(BaseResponseSchema):
    description = "Model download successful."
    body = BaseBodyResponseSchema(code=HTTPOk.code, description=description)


ModelDownload_GET_NotFoundResponseSchema = Model_GET_NotFoundResponseSchema


class Model_PUT_BodyRequestSchema(MappingSchema):
    name = SchemaNode(String(), missing=drop,
                      description="New name of the model.")
    visibility = SchemaNode(String(), missing=drop,
                            description="New visibility of the model.",
                            validator=OneOf(VISIBILITY.values()))


class Model_PUT_Endpoint(MappingSchema):
    model_uuid = model_uuid
    header = HeaderSchemaJSON()
    body = Model_PUT_BodyRequestSchema()


class Model_PUT_DataResponseSchema(MappingSchema):
    model = ModelSummaryBodyResponseSchema()


class Model_PUT_BodyResponseSchema(BaseBodyResponseSchema):
    data = Model_PUT_DataResponseSchema()


class Model_PUT_OkResponseSchema(BaseResponseSchema):
    description = "Model update successful."
    body = Model_PUT_BodyResponseSchema(code=HTTPOk.code, description=description)


class Model_PUT_BadRequestResponseSchema(BaseResponseSchema):
    description = "Model update is missing required inputs."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class Model_PUT_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Model update was refused by database."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class KeywordList(SequenceSchema):
    keyword = SchemaNode(String(), mssing=drop)


class JsonLink(MappingSchema):
    href = SchemaNode(String(), format=URL)
    rel = SchemaNode(String(), missing=drop)
    type = SchemaNode(String(), missing=drop)
    hreflang = SchemaNode(String(), missing=drop)
    title = SchemaNode(String(), missing=drop)


class Metadata(JsonLink):
    role = SchemaNode(String(), format=URL, missing=drop)
    value = SchemaNode(String(), missing=drop)


class MetadataList(SequenceSchema):
    metadata = Metadata(missing=drop)


class Format(MappingSchema):
    mimeType = SchemaNode(String())
    schema = SchemaNode(String(), missing=drop)
    encoding = SchemaNode(String(), missing=drop)


class FormatList(SequenceSchema):
    format = Format()


class InputOutputDescription(MappingSchema):
    id = SchemaNode(String(), description="Identifier of the item.")
    abstract = SchemaNode(String(), description="Item abstract.", missing=drop)
    type = SchemaNode(String(), description="Item type.", missing=drop)
    formats = FormatList(description="Item supported formats.", missing=drop)
    minOccurs = SchemaNode(Integer(), description="Minimum required instances of this item.", missing=drop)
    maxOccurs = SchemaNode(Integer(), description="Maximum instances allowed of this item.", missing=drop)


class InputOutputDescriptionList(SequenceSchema):
    input_output = InputOutputDescription(missing=drop)


class ProcessSummaryBodyResponseSchema(MappingSchema):
    uuid = SchemaNode(String(), description="Process UUID.", title="UUID")
    identifier = SchemaNode(String(), description="Process name.")
    title = SchemaNode(String(), description="Process title.")
    abstract = SchemaNode(String(), description="Process abstract.")
    keywords = KeywordList()
    metadata = MetadataList()
    version = SchemaNode(String(), description="Process version.", missing=drop)
    execute_endpoint = SchemaNode(String(), description="URL to launch a process execution.")


class ProcessDetailBodyResponseSchema(ProcessSummaryBodyResponseSchema):
    user = SchemaNode(String(), description="User UUID that launched the process.", default=None, missing=drop)
    inputs = InputOutputDescriptionList(description="Inputs of the process.")
    outputs = InputOutputDescriptionList(description="Outputs of the process.")
    limit_single_job = SchemaNode(Boolean(), description="Indicator of job limitation to a single process.")


class ProcessSummaryListSchema(SequenceSchema):
    process = ProcessSummaryBodyResponseSchema()


class Processes_GET_DataResponseSchema(MappingSchema):
    processes = ProcessSummaryListSchema()


class Processes_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Processes_GET_DataResponseSchema()


class Processes_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get processes successful."
    body = Processes_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class Processes_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Get processes by name query refused by db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class Processes_POST_BodyRequestSchema(MappingSchema):
    process_name = SchemaNode(String(), description="Name of the new process.")
    process_type = SchemaNode(String(), description="Type of the new process.")


class Processes_POST_RequestSchema(MappingSchema):
    header = HeaderSchemaJSON()
    body = Processes_POST_BodyRequestSchema()


class ProcessDataResponseSchema(MappingSchema):
    process = ProcessDetailBodyResponseSchema()


class Processes_POST_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessDataResponseSchema()


class Processes_POST_CreatedResponseSchema(BaseResponseSchema):
    description = "Create process successful."
    body = Processes_POST_BodyResponseSchema(code=HTTPCreated.code, description=description)


class Processes_POST_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to create process."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class Processes_POST_NotFoundResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve created process from db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class Processes_POST_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to add process to db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class Processes_POST_ConflictResponseSchema(BaseResponseSchema):
    description = "Process conflicts with an already existing process."
    body = ErrorBodyResponseSchema(code=HTTPConflict.code, description=description)


class ProcessEndpoint(BaseRequestSchema):
    process_uuid = process_uuid


class Process_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessDataResponseSchema()


class Process_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process successful."
    body = Process_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class Process_GET_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to retrieve process."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class Process_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve process from db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class Process_GET_NotFoundResponseSchema(BaseResponseSchema):
    description = "Process could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class ProcessJobsEndpoint(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid


class ProcessJob_GET_Endpoint(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid


class ProcessJobCurrentEndpoint(BaseRequestSchema):
    process_uuid = process_uuid


class ProcessJobLatestEndpoint(BaseRequestSchema):
    process_uuid = process_uuid


class JobTagList(SequenceSchema):
    tag = SchemaNode(String())


class InputOutput(MappingSchema):
    id = SchemaNode(String(), description="Identifier of the item.")
    value = SchemaNode(String(), description="Value of the item.", missing=drop)
    href = SchemaNode(String(), description="Reference of the item.", format=URL, missing=drop)


class InputOutputList(SequenceSchema):
    input_output = InputOutput(missing=drop)


class ProcessJobSummaryDataResponseSchema(MappingSchema):
    uuid = SchemaNode(String(), description="Job UUID.", title="UUID")
    process = SchemaNode(String(), description="Process UUID.", default=None)


class ProcessJobDetailDataResponseSchema(ProcessJobSummaryDataResponseSchema):
    task = SchemaNode(String(), description="Job sub-task UUID.")
    service = SchemaNode(String(), description="Service UUID.", default=None)
    user = SchemaNode(String(), description="User UUID that launched the process.", default=None)
    inputs = InputOutputList(description="Inputs specified on job execution.")
    status = SchemaNode(String(), description="Job status.", validator=OneOf(STATUS.values()))
    status_message = SchemaNode(String(), description="Job status message.")
    status_location = SchemaNode(String(), description="Job status full URI.", format='url')
    execute_async = SchemaNode(Boolean(), description="Asynchronous job execution specifier.")
    is_workflow = SchemaNode(Boolean(), description="Workflow job execution specifier.")
    created = SchemaNode(DateTime(), description="Job creation datetime (submission, not execution).")
    finished = SchemaNode(DateTime(), description="Job completion datetime (success or failure).")
    duration = SchemaNode(Time(), description="Job total duration between create/finished.")
    progress = SchemaNode(Integer(), description="Job percentage completion progress.")
    tags = JobTagList(description="Job execution tags.")


class ProcessJobSummaryListDataResponseSchema(SequenceSchema):
    job_summary = ProcessJobSummaryDataResponseSchema()


class ProcessJobs_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobSummaryListDataResponseSchema()


class ProcessJobs_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process jobs successful."
    body = ProcessJobs_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJobs_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve process jobs from db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class ProcessJobs_GET_NotFoundResponseSchema(BaseResponseSchema):
    description = "Process jobs could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class ProcessJob_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobDetailDataResponseSchema()


class ProcessJob_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process job successful."
    body = ProcessJob_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJob_GET_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to retrieve job."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class ProcessJob_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve process job from db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class ProcessJob_GET_NotFoundResponseSchema(BaseResponseSchema):
    description = "Process job could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class ProcessJob_PUT_BodyRequestSchema(MappingSchema):
    visibility = SchemaNode(String(), missing=drop,
                            description="New visibility of the job.",
                            validator=OneOf(VISIBILITY.values()))


class ProcessJob_PUT_Endpoint(MappingSchema):
    model_uuid = model_uuid
    header = HeaderSchemaJSON()
    body = ProcessJob_PUT_BodyRequestSchema()


class ProcessJob_PUT_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobSummaryDataResponseSchema()


class ProcessJob_PUT_OkResponseSchema(BaseResponseSchema):
    description = "Update process job successful."
    body = ProcessJob_PUT_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJob_PUT_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to update job."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class ProcessJob_PUT_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to update process job to db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


ProcessJob_PUT_NotFoundResponseSchema = ProcessJob_GET_NotFoundResponseSchema


class ProcessJobResultEndpoint(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid


class ProcessJobResultDataResponseSchema(MappingSchema):
    outputs = InputOutputList()


class ProcessJobResult_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobResultDataResponseSchema()


class ProcessJobResult_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process job result successful."
    body = Process_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJobLogsEndpoint(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid


class Logs(SequenceSchema):
    log_entry = SchemaNode(String())


class ProcessJobLogsDataResponseSchema(MappingSchema):
    logs = Logs()


class ProcessJobLogs_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobLogsDataResponseSchema()


class ProcessJobLogs_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process job logs successful."
    body = ProcessJobLogs_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJobExceptionsEndpoint(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid


class ExceptionFrameDetail(MappingSchema):
    func_name = SchemaNode(String(), description="Name of the exception frame function.")
    line_detail = SchemaNode(String(), description="Name of the exception frame function.")
    line_number = SchemaNode(String(), description="Name of the exception frame function.")
    module_name = SchemaNode(String(), description="Name of the exception frame module.")
    module_path = SchemaNode(String(), description="Path of the exception frame module.")


class Exceptions(SequenceSchema):
    exception_frame = ExceptionFrameDetail()


class ProcessJobExceptionsDataResponseSchema(MappingSchema):
    exceptions = Exceptions()


class ProcessJobExceptions_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobExceptionsDataResponseSchema()


class ProcessJobExceptions_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process job exceptions successful."
    body = ProcessJobExceptions_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJobsExecuteBodySchema(MappingSchema):
    inputs = InputOutputList(missing=drop)
    outputs = InputOutputList(missing=drop)
    # mode = SchemaNode(String(), validator=OneOf(list(execute_mode_options)))
    # response = SchemaNode(String(), validator=OneOf(list(execute_response_options)))


class ProcessJobs_POST_RequestSchema(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid
    body = ProcessJobsExecuteBodySchema()


class ProcessJobs_POST_DataResponseSchema(MappingSchema):
    job_uuid = job_uuid
    status = SchemaNode(String(), validator=OneOf(STATUS.values()))
    location = SchemaNode(String(), format=URL, description="Location of the job status for monitoring execution.")


class ProcessJobs_POST_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobs_POST_DataResponseSchema()


class ProcessJobs_POST_Headers(AcceptHeader):
    Location = SchemaNode(String(), format=URL, description="Location of the job status for monitoring execution.")


class ProcessJobs_POST_AcceptedResponseSchema(BaseResponseSchema):
    description = "Process job execute submission successful."
    header = ProcessJobs_POST_Headers()
    body = ProcessJobs_POST_BodyResponseSchema(code=HTTPAccepted.code, description=description)


class ProcessJobs_POST_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameters for process job execution."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class ProcessJobs_POST_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve process from db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class ProcessJobs_POST_NotFoundResponseSchema(BaseResponseSchema):
    description = "Process could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class Base_GET_DataResponseSchema(MappingSchema):
    docs = SchemaNode(String(), description="Information about API documentation.")
    title = SchemaNode(String(), description="API package title.")
    description = SchemaNode(String(), description="API package description.")
    version = SchemaNode(String(), description="API version string")
    url = SchemaNode(String(), description="API package source code url.")
    author = SchemaNode(String(), description="API package author.")
    email = SchemaNode(String(), description="API package email.")


class Base_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Base_GET_DataResponseSchema()


class Base_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get REST API base path successful."
    body = Base_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class SwaggerJSON_GET_OkResponseSchema(MappingSchema):
    description = SwaggerJSON.description


class SwaggerAPI_GET_OkResponseSchema(MappingSchema):
    description = "{} (this page)".format(SwaggerAPI.description)


class Versions_GET_VersionsResponseSchema(MappingSchema):
    name = SchemaNode(String(), description="Version name identifier.", example="api")
    version = SchemaNode(String(), description="Version string.", example=__meta__.__version__)
    type = SchemaNode(String(), description="Other version details.", missing=drop)


class Versions_GET_VersionListResponseSchema(SequenceSchema):
    version = Versions_GET_VersionsResponseSchema()


class Versions_GET_DataResponseSchema(MappingSchema):
    versions = Versions_GET_VersionListResponseSchema()
    db_type = SchemaNode(String(), description="Database type string.", exemple="mongodb")


class Versions_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Versions_GET_DataResponseSchema()


class Versions_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get version successful."
    body = Versions_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


# view responses
SwaggerJSON_GET_responses = {
    "200": SwaggerJSON_GET_OkResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
SwaggerAPI_GET_responses = {
    "200": SwaggerAPI_GET_OkResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Base_GET_responses = {
    "200": Base_GET_OkResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Datasets_GET_responses = {
    "200": Datasets_GET_OkResponseSchema(),
    "403": Datasets_GET_ForbiddenResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Datasets_POST_responses = {
    "201": Datasets_POST_CreatedResponseSchema(),
    "400": Datasets_POST_BadRequestResponseSchema(),
    "403": Datasets_POST_ForbiddenResponseSchema(),
    "404": Datasets_POST_NotFoundResponseSchema(),
    "409": Datasets_POST_ConflictResponseSchema(),
    "422": UnprocessableEntityResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Dataset_GET_responses = {
    "200": Dataset_GET_OkResponseSchema(),
    "400": Dataset_GET_BadRequestResponseSchema(),
    "403": Dataset_GET_ForbiddenResponseSchema(),
    "404": Dataset_GET_NotFoundResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Dataset_DELETE_responses = {
    "200": Dataset_DELETE_OkResponseSchema(),
    "400": Dataset_DELETE_BadRequestResponseSchema(),
    "404": Dataset_DELETE_NotFoundResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
DatasetDownload_GET_responses = {
    "200": DatasetDownload_GET_OkResponseSchema(),
    "404": DatasetDownload_GET_NotFoundResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Models_GET_responses = {
    "200": Models_GET_OkResponseSchema(),
    "403": Models_GET_ForbiddenResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Models_POST_responses = {
    "201": Models_POST_CreatedResponseSchema(),
    "400": Models_POST_BadRequestResponseSchema(),
    "403": Models_POST_ForbiddenResponseSchema(),
    "404": Models_POST_NotFoundResponseSchema(),
    "409": Models_POST_ConflictResponseSchema(),
    "422": Models_POST_UnprocessableEntityResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Model_GET_responses = {
    "200": Model_GET_OkResponseSchema(),
    "400": Model_GET_BadRequestResponseSchema(),
    "403": Model_GET_ForbiddenResponseSchema(),
    "404": Model_GET_NotFoundResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Model_PUT_responses = {
    "200": Model_PUT_OkResponseSchema(),
    "400": Model_PUT_BadRequestResponseSchema(),
    "403": Model_PUT_ForbiddenResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
ModelDownload_GET_responses = {
    "200": ModelDownload_GET_OkResponseSchema(),
    "404": ModelDownload_GET_NotFoundResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Processes_GET_responses = {
    "200": Processes_GET_OkResponseSchema(),
    "403": Processes_GET_ForbiddenResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Processes_POST_responses = {
    "201": Processes_POST_CreatedResponseSchema(),
    "400": Processes_POST_BadRequestResponseSchema(),
    "403": Processes_POST_ForbiddenResponseSchema(),
    "404": Processes_POST_NotFoundResponseSchema(),
    "409": Processes_POST_ConflictResponseSchema(),
    "422": UnprocessableEntityResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Process_GET_responses = {
    "200": Process_GET_OkResponseSchema(),
    "400": Process_GET_BadRequestResponseSchema(),
    "403": Process_GET_ForbiddenResponseSchema(),
    "404": Process_GET_NotFoundResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
ProcessJobs_GET_responses = {
    "200": ProcessJobs_GET_OkResponseSchema(),
    "403": ProcessJobs_GET_ForbiddenResponseSchema(),
    "404": ProcessJobs_GET_NotFoundResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
ProcessJobs_POST_responses = {
    "202": ProcessJobs_POST_AcceptedResponseSchema(),
    "400": ProcessJobs_POST_BadRequestResponseSchema(),
    "403": ProcessJobs_POST_ForbiddenResponseSchema(),
    "404": ProcessJobs_POST_NotFoundResponseSchema(),
    "422": UnprocessableEntityResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
ProcessJob_GET_responses = {
    "200": ProcessJob_GET_OkResponseSchema(),
    "400": ProcessJob_GET_BadRequestResponseSchema(),
    "403": ProcessJob_GET_ForbiddenResponseSchema(),
    "404": ProcessJob_GET_NotFoundResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
ProcessJob_PUT_responses = {
    "200": ProcessJob_PUT_OkResponseSchema(),
    "400": ProcessJob_PUT_BadRequestResponseSchema(),
    "403": ProcessJob_PUT_ForbiddenResponseSchema(),
    "404": ProcessJob_PUT_NotFoundResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
ProcessJobResult_GET_responses = {
    "200": ProcessJobResult_GET_OkResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
ProcessJobLogs_GET_responses = {
    "200": ProcessJobLogs_GET_OkResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
ProcessJobExceptions_GET_responses = {
    "200": ProcessJobExceptions_GET_OkResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}
Versions_GET_responses = {
    "200": Versions_GET_OkResponseSchema(),
    "500": InternalServerErrorResponseSchema(),
}

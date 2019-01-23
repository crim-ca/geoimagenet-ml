#!/usr/bin/env python
# coding: utf-8

from geoimagenet_ml import __meta__
from geoimagenet_ml.processes.status import job_status_values
from cornice.service import Service
from pyramid.httpexceptions import (
    HTTPOk,
    HTTPCreated,
    HTTPBadRequest,
    HTTPUnauthorized,
    HTTPForbidden,
    HTTPNotFound,
    HTTPConflict,
    HTTPUnprocessableEntity,
    HTTPInternalServerError,
)
import six
import colander


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


# Tags
APITag = 'API'
DatasetsTag = 'Datasets'
ModelsTag = 'Models'
ProcessesTag = 'Processes'


# Generic Endpoint parts
dataset_uuid = colander.SchemaNode(colander.String(), description="Dataset UUID.", title="Dataset UUID.")
model_uuid = colander.SchemaNode(colander.String(), description="Model UUID.", title="Model UUID.")
process_uuid = colander.SchemaNode(colander.String(), description="Process UUID.", title="Process UUID.")
job_uuid = colander.SchemaNode(colander.String(), description="Job UUID.", title="Job UUID.")

# Route variables
DatasetVariableUUID = "{dataset_uuid}"
ProcessVariableUUID = "{process_uuid}"
ModelVariableUUID = "{model_uuid}"
JobVariableUUID = "{job_uuid}"

# Service Routes
BaseAPI = Service(
    path='/',
    name=__meta__.__title__,
    description="GeoImageNet ML REST API information. Base path of the API.")
SwaggerJSON = Service(
    path=BaseAPI.path + 'json',
    name=__meta__.__title__ + "swagger_schemas",
    description="Schemas of {}".format(__meta__.__title__))
SwaggerAPI = Service(
    path=BaseAPI.path + 'api',
    name="swagger",
    description="Swagger of {}".format(__meta__.__title__))
DatasetsAPI = Service(
    path=BaseAPI.path + 'datasets',
    name='Datasets')
DatasetAPI = Service(
    path=BaseAPI.path + 'datasets/' + DatasetVariableUUID,
    name='Dataset')
ModelsAPI = Service(
    path=BaseAPI.path + 'models',
    name='Models')
ModelAPI = Service(
    path=BaseAPI.path + 'models/' + ModelVariableUUID,
    name='Model')
ModelDownloadAPI = Service(
    path=BaseAPI.path + 'models/' + ModelVariableUUID + '/download',
    name='ModelDownload')
ProcessesAPI = Service(
    path=BaseAPI.path + 'processes',
    name='Processes')
ProcessAPI = Service(
    path=BaseAPI.path + 'processes/' + ProcessVariableUUID,
    name='Process')
ProcessJobsAPI = Service(
    path=BaseAPI.path + 'processes/' + ProcessVariableUUID + '/jobs',
    name='ProcessJobs')
ProcessJobAPI = Service(
    path=BaseAPI.path + 'processes/' + ProcessVariableUUID + '/jobs/' + JobVariableUUID,
    name='ProcessJob')
ProcessJobResultAPI = Service(
    path=BaseAPI.path + 'processes/' + ProcessVariableUUID + '/jobs/' + JobVariableUUID + '/result',
    name='ProcessJobResult')
ProcessJobLogsAPI = Service(
    path=BaseAPI.path + 'processes/' + ProcessVariableUUID + '/jobs/' + JobVariableUUID + '/logs',
    name='ProcessJobLogs')
ProcessJobExceptionsAPI = Service(
    path=BaseAPI.path + 'processes/' + ProcessVariableUUID + '/jobs/' + JobVariableUUID + '/exceptions',
    name='ProcessJobExceptions')
VersionsAPI = Service(
    path=BaseAPI.path + 'versions',
    name='Versions')


# Security
SecurityDefinitionAPI = {'securityDefinitions': {'cookieAuth': {'type': 'apiKey', 'in': 'cookie', 'name': 'auth_tkt'}}}
SecurityAdministratorAPI = [{'cookieAuth': []}]
SecurityEveryoneAPI = []

# Content
ContentTypeJSON = 'application/json'
ContentTypeHTML = 'text/html'


# Service Routes Utility
def service_api_route_info(service_api):
    return {'name': service_api.name, 'pattern': service_api.path}


def get_security(service, method):
    definitions = service.definitions
    args = {}
    for definition in definitions:
        met, view, args = definition
        if met == method:
            break
    return SecurityAdministratorAPI if 'security' not in args else args['security']


class HeaderSchemaJSON(colander.MappingSchema):
    content_type = colander.SchemaNode(colander.String(), example=ContentTypeJSON, default=ContentTypeJSON)
    content_type.name = 'Content-Type'


class HeaderSchemaHTML(colander.MappingSchema):
    content_type = colander.SchemaNode(colander.String(), example=ContentTypeHTML, default=ContentTypeHTML)
    content_type.name = 'Content-Type'


class AcceptHeader(colander.MappingSchema):
    Accept = colander.SchemaNode(colander.String(), missing=colander.drop, default=ContentTypeJSON,
                                 validator=colander.OneOf([ContentTypeJSON, ContentTypeHTML]))


class BaseRequestSchema(colander.MappingSchema):
    header = AcceptHeader()


class BaseMetaResponseSchema(colander.MappingSchema):
    code = colander.SchemaNode(
        colander.Integer(), description="HTTP response code.", example=HTTPOk.code)
    type = colander.SchemaNode(
        colander.String(), description="Response content type.", example="application/json")
    detail = colander.SchemaNode(
        colander.String(), description="Response status message.")
    route = colander.SchemaNode(
        colander.String(), description="Request route called that generated the response.", missing=colander.drop)
    uri = colander.SchemaNode(
        colander.String(), description="Request URI that generated the response.", missing=colander.drop)
    method = colander.SchemaNode(
        colander.String(), description="Request method that generated the response.", missing=colander.drop)


class BaseBodyResponseSchema(colander.MappingSchema):
    meta = BaseMetaResponseSchema()
    data = colander.MappingSchema(default={})

    __code = None
    __desc = None

    def __init__(self, code, description):
        super(BaseBodyResponseSchema, self).__init__()
        assert isinstance(code, int)
        assert isinstance(description, six.string_types)
        self.__code = code
        self.__desc = description

        # update the values
        child_nodes = getattr(self, 'children')
        for node in child_nodes:
            if node.name == 'meta':
                for meta_node in getattr(node, 'children'):
                    if meta_node.name == 'code':
                        meta_node.example = self.__code
                    if meta_node.name == 'detail':
                        meta_node.example = self.__desc


class BaseResponseSchema(colander.MappingSchema):
    description = 'UNDEFINED'
    header = AcceptHeader()
    body = BaseBodyResponseSchema(code=HTTPOk.code, description=description)


class ErrorBodyResponseSchema(BaseBodyResponseSchema):
    data = colander.MappingSchema()


class UnauthorizedDataResponseSchema(colander.MappingSchema):
    route_name = colander.SchemaNode(colander.String(), description="Specified route")
    request_url = colander.SchemaNode(colander.String(), description="Specified url")


class UnauthorizedResponseSchema(BaseResponseSchema):
    description = "Unauthorized. Insufficient user privileges or missing authentication headers."
    body = ErrorBodyResponseSchema(code=HTTPUnauthorized.code, description=description)


class NotFoundResponseSchema(BaseResponseSchema):
    description = "The route resource could not be found."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class UnprocessableEntityResponseSchema(BaseResponseSchema):
    description = "Invalid value specified."
    body = ErrorBodyResponseSchema(code=HTTPUnprocessableEntity.code, description=description)


class TracebackListSchema(colander.SequenceSchema):
    item = colander.SchemaNode(colander.String(), missing=colander.drop,
                               description="Summary line of the traceback.")


class InternalServerErrorDataSchema(colander.MappingSchema):
    exception = colander.SchemaNode(colander.String(), missing=colander.drop,
                                    description="Exception message description.")
    traceback = TracebackListSchema(default=[], missing=colander.drop,
                                    description="Exception stack trace caused by the request.")
    caller = colander.MappingSchema(default={}, missing=colander.drop,
                                    description="Details of the calling request generating this error.")


class InternalServerErrorBodySchema(ErrorBodyResponseSchema):
    def __init__(self, description):
        super(InternalServerErrorBodySchema, self).\
            __init__(code=HTTPInternalServerError.code, description=description)

    error = InternalServerErrorDataSchema(description="Details of the generated error.")


class InternalServerErrorResponseSchema(BaseResponseSchema):
    description = "Internal Server Error. Unhandled exception occurred."
    body = InternalServerErrorBodySchema(description=description)


class DatasetBodyResponseSchema(colander.MappingSchema):
    uuid = colander.SchemaNode(colander.String(), description="Dataset uuid.", title="UUID")
    name = colander.SchemaNode(colander.String(), description="Dataset name.")
    path = colander.SchemaNode(colander.String(), description="Dataset path.")


class DatasetNamesListSchema(colander.SequenceSchema):
    item = DatasetBodyResponseSchema()


class Datasets_GET_DataResponseSchema(colander.MappingSchema):
    datasets = DatasetNamesListSchema()


class Datasets_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Datasets_GET_DataResponseSchema()


class Datasets_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get datasets successful."
    body = Datasets_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class Datasets_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Get datasets by name query refused by db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class DatasetDataResponseSchema(colander.MappingSchema):
    dataset = DatasetBodyResponseSchema()


class Datasets_POST_BodyRequestSchema(colander.MappingSchema):
    dataset_name = colander.SchemaNode(colander.String(), description="Name of the new dataset.")
    dataset_path = colander.SchemaNode(colander.String(), description="Path of the new dataset.")


class Datasets_POST_RequestSchema(colander.MappingSchema):
    header = HeaderSchemaJSON()
    body = Datasets_POST_BodyRequestSchema()


class Datasets_POST_BodyResponseSchema(BaseBodyResponseSchema):
    data = DatasetDataResponseSchema()


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


class Dataset_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = DatasetDataResponseSchema()


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


class ModelBodyResponseSchema(colander.MappingSchema):
    uuid = colander.SchemaNode(colander.String(), description="Model uuid.", title="UUID")
    name = colander.SchemaNode(colander.String(), description="Model name.")
    path = colander.SchemaNode(colander.String(), description="Model path.")


class ModelNamesListSchema(colander.SequenceSchema):
    item = ModelBodyResponseSchema()


class Models_GET_DataResponseSchema(colander.MappingSchema):
    models = ModelNamesListSchema()


class Models_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Models_GET_DataResponseSchema()


class Models_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get models successful."
    body = Models_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class Models_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Get models by name query refused by db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class ModelDataResponseSchema(colander.MappingSchema):
    model = ModelBodyResponseSchema()


class Models_POST_BodyRequestSchema(colander.MappingSchema):
    model_name = colander.SchemaNode(colander.String(), description="Name of the new model.")
    model_path = colander.SchemaNode(colander.String(), description="Path of the new model.")


class Models_POST_RequestSchema(colander.MappingSchema):
    header = HeaderSchemaJSON()
    body = Models_POST_BodyRequestSchema()


class Models_POST_BodyResponseSchema(BaseBodyResponseSchema):
    data = ModelDataResponseSchema()


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


class ModelEndpoint(BaseRequestSchema):
    model_uuid = model_uuid


class Model_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ModelDataResponseSchema()


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


class ModelDownload_GET_NotFoundResponseSchema(BaseResponseSchema):
    description = "Model download file could not be found."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class ProcessBodyResponseSchema(colander.MappingSchema):
    uuid = colander.SchemaNode(colander.String(), description="Process uuid.", title="UUID")
    name = colander.SchemaNode(colander.String(), description="Process name.")


class ProcessNamesListSchema(colander.SequenceSchema):
    item = ProcessBodyResponseSchema()


class Processes_GET_DataResponseSchema(colander.MappingSchema):
    processes = ProcessNamesListSchema()


class Processes_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Processes_GET_DataResponseSchema()


class Processes_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get processes successful."
    body = Processes_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class Processes_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Get processes by name query refused by db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class ProcessDataResponseSchema(colander.MappingSchema):
    process = ProcessBodyResponseSchema()


class Processes_POST_BodyRequestSchema(colander.MappingSchema):
    process_name = colander.SchemaNode(colander.String(), description="Name of the new process.")


class Processes_POST_RequestSchema(colander.MappingSchema):
    header = HeaderSchemaJSON()
    body = Processes_POST_BodyRequestSchema()


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


class ProcessJobs_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process jobs successful."
    body = Process_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJobs_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve process jobs from db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class ProcessJobs_GET_NotFoundResponseSchema(BaseResponseSchema):
    description = "Process jobs could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class InputOutput(colander.MappingSchema):
    id = colander.SchemaNode(colander.String(), description="Identifier of the item.")
    value = colander.SchemaNode(colander.String(), description="Value of the item.", missing=colander.drop)
    href = colander.SchemaNode(colander.String(), description="Reference of the item.", missing=colander.drop)


class InputOutputList(colander.SequenceSchema):
    item = InputOutput(missing=colander.drop)


class ProcessJobEndpoint(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid


class Tags(colander.SequenceSchema):
    tag = colander.SchemaNode(colander.String())


class ProcessJobDataResponseSchema(colander.MappingSchema):
    uuid = colander.SchemaNode(colander.String(), description="Job uuid.", title="UUID")
    task_uuid = colander.SchemaNode(colander.String(), description="Job sub-task UUID.")
    service_uuid = colander.SchemaNode(colander.String(), description="Service UUID.", default=None)
    process_uuid = colander.SchemaNode(colander.String(), description="Process UUID.", default=None)
    user_uuid = colander.SchemaNode(colander.String(), description="User UUID that launched the process.", default=None)
    inputs = InputOutputList(description="Inputs specified on job execution.")
    status = colander.SchemaNode(colander.String(), description="Job status.",
                                 validator=colander.OneOf(list(job_status_values)))
    status_message = colander.SchemaNode(colander.String(), description="Job status message.")
    status_location = colander.SchemaNode(colander.String(), description="Job status full URI.", format='url')
    execute_async = colander.SchemaNode(colander.Boolean(), description="Asynchronous job execution specifier.")
    is_workflow = colander.SchemaNode(colander.Boolean(), description="Workflow job execution specifier.")
    created = colander.SchemaNode(colander.DateTime(), description="Job creation datetime (submission, not execution).")
    finished = colander.SchemaNode(colander.DateTime(), description="Job completion datetime (success or failure).")
    duration = colander.SchemaNode(colander.Time(), description="Job total duration between create/finished.")
    progress = colander.SchemaNode(colander.Integer(), description="Job percentage completion progress.")
    tags = Tags(description="Job execution tags.")


class ProcessJob_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobDataResponseSchema()


class ProcessJob_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process job successful."
    body = ProcessJob_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJob_GET_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameter specified to retrieve job."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class ProcessJob_GET_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve process job from db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class ProcessJob_GET_NotFoundResponseSchema(BaseResponseSchema):
    description = "Process job could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class ProcessJobResultEndpoint(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid


class ProcessJobResultDataResponseSchema(colander.MappingSchema):
    outputs = InputOutputList()


class ProcessJobResult_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobResultDataResponseSchema()


class ProcessJobResult_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process job result successful."
    body = Process_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJobLogsEndpoint(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid


class Logs(colander.SequenceSchema):
    item = colander.SchemaNode(colander.String())


class ProcessJobLogsDataResponseSchema(colander.MappingSchema):
    logs = Logs()


class ProcessJobLogs_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobLogsDataResponseSchema()


class ProcessJobLogs_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process job logs successful."
    body = ProcessJobLogs_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJobExceptionsEndpoint(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid


class ExceptionFrameDetail(colander.MappingSchema):
    func_name = colander.SchemaNode(colander.String(), description="Name of the exception frame function.")
    line_detail = colander.SchemaNode(colander.String(), description="Name of the exception frame function.")
    line_number = colander.SchemaNode(colander.String(), description="Name of the exception frame function.")
    module_name = colander.SchemaNode(colander.String(), description="Name of the exception frame module.")
    module_path = colander.SchemaNode(colander.String(), description="Path of the exception frame module.")


class Exceptions(colander.SequenceSchema):
    item = ExceptionFrameDetail()


class ProcessJobExceptionsDataResponseSchema(colander.MappingSchema):
    exceptions = Exceptions()


class ProcessJobExceptions_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = ProcessJobExceptionsDataResponseSchema()


class ProcessJobExceptions_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get process job exceptions successful."
    body = ProcessJobExceptions_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJobsExecuteBodySchema(colander.MappingSchema):
    inputs = InputOutputList(missing=colander.drop)
    outputs = InputOutputList(missing=colander.drop)
    # mode = SchemaNode(String(), validator=colander.OneOf(list(execute_mode_options)))
    # response = SchemaNode(String(), validator=colander.OneOf(list(execute_response_options)))


class ProcessJobs_POST_RequestSchema(BaseRequestSchema):
    process_uuid = process_uuid
    job_uuid = job_uuid
    body = ProcessJobsExecuteBodySchema()


class ProcessJobs_POST_OkResponseSchema(BaseResponseSchema):
    description = "Process job execute submission successful."
    body = Process_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class ProcessJobs_POST_BadRequestResponseSchema(BaseResponseSchema):
    description = "Invalid parameters for process job execution."
    body = ErrorBodyResponseSchema(code=HTTPBadRequest.code, description=description)


class ProcessJobs_POST_ForbiddenResponseSchema(BaseResponseSchema):
    description = "Failed to retrieve process from db."
    body = ErrorBodyResponseSchema(code=HTTPForbidden.code, description=description)


class ProcessJobs_POST_NotFoundResponseSchema(BaseResponseSchema):
    description = "Process could not be found in db."
    body = ErrorBodyResponseSchema(code=HTTPNotFound.code, description=description)


class Base_GET_DataResponseSchema(colander.MappingSchema):
    docs = colander.SchemaNode(colander.String(), description="Information about API documentation.")
    title = colander.SchemaNode(colander.String(), description="API package title.")
    description = colander.SchemaNode(colander.String(), description="API package description.")
    version = colander.SchemaNode(colander.String(), description="API version string")
    url = colander.SchemaNode(colander.String(), description="API package source code url.")
    author = colander.SchemaNode(colander.String(), description="API package author.")
    email = colander.SchemaNode(colander.String(), description="API package email.")


class Base_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Base_GET_DataResponseSchema()


class Base_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get REST API base path successful."
    body = Base_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


class SwaggerJSON_GET_OkResponseSchema(colander.MappingSchema):
    description = SwaggerJSON.description


class SwaggerAPI_GET_OkResponseSchema(colander.MappingSchema):
    description = "{} (this page)".format(SwaggerAPI.description)


class Versions_GET_VersionsResponseSchema(colander.MappingSchema):
    name = colander.SchemaNode(colander.String(), description="Version name identifier.", example="api")
    version = colander.SchemaNode(colander.String(), description="Version string.",
                                  example=__meta__.__version__)
    type = colander.SchemaNode(colander.String(), description="Other version details.", missing=colander.drop)


class Versions_GET_VersionListResponseSchema(colander.SequenceSchema):
    version = Versions_GET_VersionsResponseSchema()


class Versions_GET_DataResponseSchema(colander.MappingSchema):
    versions = Versions_GET_VersionListResponseSchema()
    db_type = colander.SchemaNode(colander.String(), description="Database type string.", exemple="mongodb")


class Versions_GET_BodyResponseSchema(BaseBodyResponseSchema):
    data = Versions_GET_DataResponseSchema()


class Versions_GET_OkResponseSchema(BaseResponseSchema):
    description = "Get version successful."
    body = Versions_GET_BodyResponseSchema(code=HTTPOk.code, description=description)


# view responses
SwaggerJSON_GET_responses = {
    '200': SwaggerJSON_GET_OkResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
SwaggerAPI_GET_responses = {
    '200': SwaggerAPI_GET_OkResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Base_GET_responses = {
    '200': Base_GET_OkResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Datasets_GET_responses = {
    '200': Datasets_GET_OkResponseSchema(),
    '403': Datasets_GET_ForbiddenResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Datasets_POST_responses = {
    '201': Datasets_POST_CreatedResponseSchema(),
    '400': Datasets_POST_BadRequestResponseSchema(),
    '403': Datasets_POST_ForbiddenResponseSchema(),
    '404': Datasets_POST_NotFoundResponseSchema(),
    '409': Datasets_POST_ConflictResponseSchema(),
    '422': UnprocessableEntityResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Dataset_GET_responses = {
    '200': Dataset_GET_OkResponseSchema(),
    '400': Dataset_GET_BadRequestResponseSchema(),
    '403': Dataset_GET_ForbiddenResponseSchema(),
    '404': Dataset_GET_NotFoundResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Models_GET_responses = {
    '200': Models_GET_OkResponseSchema(),
    '403': Models_GET_ForbiddenResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Models_POST_responses = {
    '201': Models_POST_CreatedResponseSchema(),
    '400': Models_POST_BadRequestResponseSchema(),
    '403': Models_POST_ForbiddenResponseSchema(),
    '404': Models_POST_NotFoundResponseSchema(),
    '409': Models_POST_ConflictResponseSchema(),
    '422': Models_POST_UnprocessableEntityResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Model_GET_responses = {
    '200': Model_GET_OkResponseSchema(),
    '400': Model_GET_BadRequestResponseSchema(),
    '403': Model_GET_ForbiddenResponseSchema(),
    '404': Model_GET_NotFoundResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
ModelDownload_GET_responses = {
    '200': ModelDownload_GET_OkResponseSchema(),
    '404': ModelDownload_GET_NotFoundResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Processes_GET_responses = {
    '200': Processes_GET_OkResponseSchema(),
    '403': Processes_GET_ForbiddenResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Processes_POST_responses = {
    '201': Processes_POST_CreatedResponseSchema(),
    '400': Processes_POST_BadRequestResponseSchema(),
    '403': Processes_POST_ForbiddenResponseSchema(),
    '404': Processes_POST_NotFoundResponseSchema(),
    '409': Processes_POST_ConflictResponseSchema(),
    '422': UnprocessableEntityResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Process_GET_responses = {
    '200': Process_GET_OkResponseSchema(),
    '400': Process_GET_BadRequestResponseSchema(),
    '403': Process_GET_ForbiddenResponseSchema(),
    '404': Process_GET_NotFoundResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
ProcessJobs_GET_responses = {
    '200': ProcessJobs_GET_OkResponseSchema(),
    '403': ProcessJobs_GET_ForbiddenResponseSchema(),
    '404': ProcessJobs_GET_NotFoundResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
ProcessJobs_POST_responses = {
    '200': ProcessJobs_POST_OkResponseSchema(),
    '400': ProcessJobs_POST_BadRequestResponseSchema(),
    '403': ProcessJobs_POST_ForbiddenResponseSchema(),
    '404': ProcessJobs_POST_NotFoundResponseSchema(),
    '422': UnprocessableEntityResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
ProcessJob_GET_responses = {
    '200': ProcessJob_GET_OkResponseSchema(),
    '400': ProcessJob_GET_BadRequestResponseSchema(),
    '403': ProcessJob_GET_ForbiddenResponseSchema(),
    '404': ProcessJob_GET_NotFoundResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
ProcessJobResult_GET_responses = {
    '200': ProcessJobResult_GET_OkResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
ProcessJobLogs_GET_responses = {
    '200': ProcessJobLogs_GET_OkResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
ProcessJobExceptions_GET_responses = {
    '200': ProcessJobExceptions_GET_OkResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}
Versions_GET_responses = {
    '200': Versions_GET_OkResponseSchema(),
    '500': InternalServerErrorResponseSchema(),
}

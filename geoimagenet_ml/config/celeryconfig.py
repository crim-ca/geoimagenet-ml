# celery scheduler config
# http://celery.readthedocs.org/en/latest/configuration.html
import os

MONGODB_HOST = str(os.getenv('MONGODB_HOST', 'mongodb'))
MONGODB_PORT = str(os.getenv('MONGODB_PORT', 27017))

# Broker settings.
BROKER_URL = 'mongodb://{host}:{port}/celery'.format(host=MONGODB_HOST, port=MONGODB_PORT)

# Backend to store task state and results.
CELERY_RESULT_BACKEND = 'mongodb://{host}:{port}'.format(host=MONGODB_HOST, port=MONGODB_PORT)

# Backend to store task state and results.
CELERY_MONGODB_BACKEND_SETTINGS = {'database': 'celery', 'taskmeta_collection': 'celery_tasks', }

# Allow creation of pid/log directories and user/group ownership by celery
CELERY_CREATE_DIRS = 1

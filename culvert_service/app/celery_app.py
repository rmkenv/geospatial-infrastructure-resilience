import os
from celery import Celery

# Use environment variables for broker and backend URLs, with defaults for local development
broker_url = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")
result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/0")

# Initialize the Celery app
celery = Celery(
    'tasks',
    broker=broker_url,
    backend=result_backend,
    include=['app.tasks']  # Explicitly include the tasks module
)

# Optional configuration
celery.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)

if __name__ == '__main__':
    celery.start()

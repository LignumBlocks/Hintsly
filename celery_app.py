from celery import Celery

celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.task_routes = {
    'tasks.verify_hacks_from_text': {'queue': 'hacks'}
}
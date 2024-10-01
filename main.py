from fastapi import FastAPI
from tasks import verify_hacks_task
from celery.result import AsyncResult

app = FastAPI()

@app.post("/verify-hack/")
async def verify_hack(text: str):
    task = verify_hacks_task.delay(text)
    return {"task_id": task.id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    task_result = AsyncResult(task_id)
    if task_result.state == "PENDING":
        return {"status": "pending"}
    return {"status": task_result.state, "result": task_result.result}

from typing_extensions import DefaultDict, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import time
import asyncio
import uuid
import queue
import logging


@dataclass
class Task:
    id_:int
    execution_time:int
    caption:str

@dataclass
class IndexTask:
    task_id:int
    link:str

@dataclass
class Item:
    id: str
    name: str
    

fake_db = {}
app = FastAPI()
task_queue = queue.SimpleQueue()
TASK_ID = 1


'''
async def process_requests(q: asyncio.Queue, pool: ProcessPoolExecutor):
    while True:
        item = await q.get()  # Get a request from the queue
        loop = asyncio.get_running_loop()
        fake_db[item.id] = 'Processing...'
        r = await loop.run_in_executor(pool, cpu_bound_task, item)
        q.task_done()  # tell the queue that the processing on the task is completed
        fake_db[item.id] = 'Done.'


@asynccontextmanager
async def lifespan(app: FastAPI):
    q = asyncio.Queue()  # note that asyncio.Queue() is not thread safe
    pool = ProcessPoolExecutor()
    asyncio.create_task(process_requests(q, pool))  # Start the requests processing task
    yield {'q': q, 'pool': pool}
    pool.shutdown()  # free any resources that the pool is using when the currently pending futures are done executing

'''
def process_texts(texts:DefaultDict[str, str]) -> str:
    return ''


@app.post("/push_last_task")
async def get_llava_task(doneTask:Task):
    '''
    '''
    fake_db[doneTask.id_] = doneTask.caption
    return JSONResponse('Task done', status_code=200)


@app.get("/get_task")
async def get_last_task():
    if task_queue.empty():
        return -1 
    return JSONResponse(content=task_queue.get(), status_code=200)  

@app.post("/add_task/{link:path}")
async def add_index_task(link:str):

    global TASK_ID
    task = IndexTask(task_id=TASK_ID,link=link)
    TASK_ID +=1
    task_queue.put(task)
    logging.getLogger(__name__).warn(f" added {task}")
    return TASK_ID-1  
    
'''
@app.get("/status")
async def check_status(item_id: str):
    if item_id in fake_db:
        return {'status': fake_db[item_id]}
    else:
        return JSONResponse("Item ID Not Found", status_code=404)
'''

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)

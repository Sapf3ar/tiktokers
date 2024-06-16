from fastapi import FastAPI
import pandas as pd
import faiss
import numpy as np
from fastapi.responses import JSONResponse
from dataclasses import dataclass, asdict
import queue
import logging
from stt import process_video_stt
from text_extractor import TextExtractor
from api import summary_modalities, call_vllm_api
from sentence_transformers import SentenceTransformer

import urllib.request
from urllib.parse import urlparse
import os


@dataclass
class Task:
    id_:int
    execution_time:int
    video_link:str
    caption:str
    error:str

@dataclass
class IndexTask:
    task_id:int
    link:str
    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class Query:
    query:str


@dataclass
class Item:
    id: str
    name: str
    

fake_db = {}
app = FastAPI()
embedder = SentenceTransformer('intfloat/multilingual-e5-base')
task_queue = queue.SimpleQueue()
text_extactor = TextExtractor()
total_data = pd.read_csv("13k_with_summary.csv")

index = faiss.read_index("data/faiss.index")
TASK_ID = 1
VLLM_MODEL_NAME = "~/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/359ec69a0f92259a3cd2da3bb01d31e16c260cfc/"


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
async def process_video(video_link:str, caption: str):
    video_path = urlparse(video_link).path.split("/")[-1]
    urllib.request.urlretrieve(video_link, local_video_path := os.path.join("videos", video_path))

    stt_result = " "
    stt_result = await process_video_stt(local_video_path)

    ocr_result = ""
    try: 
        ocr_result = text_extactor.extract_text(local_video_path) 
    except Exception as e:
        logging.warning(str(e) + ' in ocr')

    omniprompt = summary_modalities(ocr=ocr_result,
               asr=stt_result,
               caption=caption,
               call_llm=call_vllm_api,
               model_path=VLLM_MODEL_NAME
               )
    row_dict = {key:np.nan for key in total_data.columns}
    row_dict['link'] = video_link
    row_dict['ocr'] = ocr_result
    row_dict['llava'] = caption
    row_dict['omnisummary'] = omniprompt
    row_dict['asr'] = stt_result
    total_data._append(row_dict, ignore_index=True)
    

    index.add(embedder.encode([omniprompt]))

@app.post("/push_last_task")
async def get_llava_task(doneTask:Task):
    '''
    '''
    logging.getLogger(__name__).warning(doneTask)
    fake_db[doneTask.id_] = {"caption" : doneTask.caption,
                             "error": doneTask.error,
                             "execution_time":doneTask.execution_time}
    await process_video(doneTask.video_link, 
                  caption=doneTask.caption)
    if doneTask.error:
        return JSONResponse(f"Task {doneTask.id_} exited with {doneTask.error}")
    return JSONResponse('Task done', status_code=200)

@app.get("/get_task_status")
async def get_task_status(task_id:int):
    error = "Task not found" 
    if task_id in fake_db.keys():
        error = "Task finished with " + fake_db[task_id]
    return JSONResponse(content=error, status_code=200)  

@app.get("/get_task")
async def get_last_task():
    if task_queue.empty():
        return -1 
    return JSONResponse(content=task_queue.get().dict(), status_code=200)  

@app.post("/add_task/{link:path}")
async def add_index_task(link:str):

    global TASK_ID
    task = IndexTask(task_id=TASK_ID,link=link)
    TASK_ID +=1
    task_queue.put(task)
    logging.getLogger(__name__).warn(f" added {task}")
    return TASK_ID-1  
    

@app.post("/search")
async def search_index(query:Query):
    emb = embedder.encode([query.query])
    _, indexes = index.search(emb, 10)
    logging.getLogger(__name__).warning(f"Finished search for {query}")
    links = total_data.iloc[indexes].link
    return JSONResponse(content={str(i):links[i] for i in range(len(links))}, status_code=200)
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

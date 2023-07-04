"""クライアントからの複数リクエストをバッチ化してNNを実行させるサーバー
"""
import os
import sys


# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..']) # p2_ct_project
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

import asyncio
import itertools
import functools
from sanic import Sanic
from sanic.response import (
    json,
    text,
)
from sanic.log import logger
from sanic.exceptions import ServerError

import sanic
import threading
import PIL.Image
import io
import torch
import torchvision

app = Sanic(__name__)

device = torch.device('cpu')

# we only run 1 interface run at any time (one could schedule between several runners if desired)
MAX_QUEUE_SIZE = 3 # we accept a backlog of MAX_QUEUE_SIZE before handing out "Too busy" errors
MAX_BATCH_SIZE = 2 # we put at most MAX_BATCH_SIZE things in a single batch
MAX_WAIT = 1 # we wait at most MAX_WAIT seconds before running for more inputs to arrive in batching

class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg

class ModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        self.queue = []

        self.queue_lock = None

        self.model = get_pretrained_model(self.model_name,
                                          map_location=device,
                                          )
        
        self.needs_processing = None
        self.needs_processing_timer = None

    # 推論処理のスケジュール設定
    def schedule_processing_if_needed(self):
        if len(self.queue) >= MAX_BATCH_SIZE:
            logger.debug("next batch ready when processing a batch")
            # 即時, 推論処理を実行するトリガーをセット
            self.needs_processing.set() # Event ON

        elif self.queue:
            logger.debug("queue noempty when processing a batch, setting next timer")
            # 次の指定時間が経過したら, 推論処理を実行するトリガーをタイマーにセット
            self.needs_processing_timer = app.loop.call_at(
                self.queue[0]['time'] + MAX_WAIT, # when
                self.needs_processing.set, # callback (Event ON)
                )
    
    async def process_input(self, 
                            input,
                            ):
        """リクエストされた内容をデコードして,
        入力データをキューに溜めて, 処理が完了するのを待つ

        Args:
            input (_type_): _description_

        Raises:
            HandlingError: _description_

        Returns:
            _type_: _description_
        """
        our_task = {
            "done_event": asyncio.Event(loop=app.loop),
            "input": input,
            "time": app.loop.time(),
        }


        async with self.queue_lock:
            if len(self.queue) >= MAX_QUEUE_SIZE:
                raise HandlingError("I'm too busy", code=503)
            self.queue.append(our_task) # タスクの追加
            logger.debug("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed()

        await our_task["done_event"].wait() # イベントがONになるまで待機
        return our_task["output"]
    
    # ワーカースレッドで実行
    def run_model(self,
                  batch,
                  ):
        return self.model(batch.to(device)).to('cpu')
    
    async def model_runner(self):
        """永遠に実行される
        モデルを実行するタイミングで入力データのミニバッチを組み立て,
        (プライマリースレッドで他の内容が処理できるように)ワーカースレッドで
        モデルを実行して結果を返す.
        """
        self.queue_lock = asyncio.Lock(loop=app.loop)
        self.needs_processing = asyncio.Event(loop=app.loop)
        logger.info("started model runner for {}".format(self.model_name))

        while True:
            await self.needs_processing.wait() # イベントがONになるまで待機
            self.needs_processing.clear() # Event OFF

            if self.needs_processing_timer is not None:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None
            
            async with self.queue_lock:
                if self.queue:
                    longest_wait = app.loop.time() - self.queue[0]['time']
                else:
                    longest_wait = None
                
                logger.debug("launching processing. queue size: {}. longest wait: {}".format(len(self.queue), longest_wait))

                # キューに溜まっている全データでバッチ推論を実行
                to_process = self.queue[:MAX_BATCH_SIZE]
                del self.queue[:len(to_process)]

                self.schedule_processing_if_needed()

            # so here we copy, it would be neater to avoid this
            batch = torch.stack([t['input'] for t in to_process], dim=0)
            # we could delete inputs here...


            result = await app.loop.run_in_executor(
                None,
                functools.partial(self.run_model, batch)
            )

            for t, r in zip(to_process, result):
                t['output'] = r
                t["done_event"].set()
            del to_process

style_transfer_runner = ModelRunner(sys.argv[1])

@app.route('/image', methods=['PUT'], stream=True)
async def image(request):
    try:
        print(request.headers)
        content_length = int(request.headers.get('content-length', '0'))
        MAX_SIZE = 2*22 # 10MB

        if content_length:
            if content_length > MAX_SIZE:
                raise HandlingError("Too large")
            data = bytearray(content_length)
        else:
            data = bytearray(MAX_SIZE)
        
        pos = 0

        while True:
            # so this still copies too much stuff.
            data_part = await request.stream.read()
            if data_part is None:
                break
            data[pos : len(data_part) + pos] = data_part
            pos += len(data_part)
            if pos > MAX_SIZE:
                raise HandlingError("Too large")
            
        # ideally, we would minimize preprocessing...
        im = PIL.Image.open(io.BytesIO(data))
        im = torchvision.transforms.functional.resize(im, (228, 228))
        im = torchvision.transforms.functional.to_tensor(im)
        im = im[:3] # drop alpha channels if present
        if im.dim() != 3 or im.size(0) < 3 or im.size(0) > 4:
            raise HandlingError("need rgb image")
        out_im = await style_transfer_runner.process_input(im)
        out_im = torchvision.transforms.functional.to_pil_image(out_im)
        imgByteArr = io.BytesIO()
        out_im.save(imgByteArr, format="JPEG")
        return sanic.response.raw(imgByteArr.getvalue(), 
                                  status=200, 
                                  content_type='image/jpeg',
                                  )
    
    except HandlingError as e:
        # we don't want these to be logged...
        return sanic.response.text(e.handling_mgs, status=e.handling_code)
    
app.add_task(style_transfer_runner.model_runner())
app.run(host="0.0.0.0", port=8000, debug=True)


            

        
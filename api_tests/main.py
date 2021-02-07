from PIL import Image
import requests
import os
from io import BytesIO
import base64
import time
import threading
import queue

WAIT_TIME_IN_SECONDS = 1 #Time in seconds to wait before next api call
THREAD_COUNT = 15
IMAGE_DIRECTORY = "images"

recorded_times = []
total_execution_time = 0
failure_count = 0
main_queue = queue.Queue()
queue_size = 0

# convert PIL image to base64

def image_to_base64(image: Image.Image, image_format="JPEG") -> str:
    memory_buffer = BytesIO()
    image.save(memory_buffer, format=image_format)
    return base64.b64encode(memory_buffer.getvalue()).decode("ascii")

# send image to api and measure response

def image_to_api(image: Image.Image):
    base_64_image = image_to_base64(image, image.format)
    print("Sending ...")
    start = time.time()

    global failure_count

    try:
      response = requests.post(
        "http://localhost:8000/api/segment_article",
          json={"image_base64": base_64_image}
      )

      if(response.status_code < 200 or response.status_code > 299):
        failure_count += 1
            
      print(response.content)

    except:
        print("Something went wrong")

    end = time.time()
    recorded_times.append(end - start)

    global total_execution_time
    total_execution_time += end - start

    time.sleep(WAIT_TIME_IN_SECONDS)

# get all images in directory

def getListOfFiles(dirName):
  global queue_size
  # create a list of file and sub directories
  # names in the given directory
  listOfFile = os.listdir(dirName)
   # Iterate over all the entries
  for entry in listOfFile:
    # Create full path
    fullPath = os.path.join(dirName, entry)
    # If entry is a directory then get the list of files in this directory
    if os.path.isdir(fullPath):
      getListOfFiles(fullPath)
    else:
      try:
        pil_image = Image.open(fullPath)
        if pil_image is not None:
          main_queue.put_nowait(pil_image)
          queue_size += 1
      except:
        print("Not image")


getListOfFiles(IMAGE_DIRECTORY)

# worker for threading

class Worker(threading.Thread):
    def __init__(self, q, *args, **kwargs):
        self.q = q
        super().__init__(*args, **kwargs)

    def run(self):
        while True:
            try:
                work = self.q.get(timeout=3)  # 3s timeout
                image_to_api(work)
            except queue.Empty:
                return
            # do whatever work you have to do on work
            self.q.task_done()


for _ in range(THREAD_COUNT):
    Worker(main_queue).start()
main_queue.join()

average_time = sum(recorded_times)/len(recorded_times)

print(f"\nFailure count is: {failure_count}\n")
print(f"\nFailure rate is: {(failure_count/queue_size) * 100}\n")
print(f"\nAverage api response time is: {average_time}\n")
print(f"\nTotal api response time is: {total_execution_time}\n")

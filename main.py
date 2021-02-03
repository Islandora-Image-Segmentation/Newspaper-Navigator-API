import uvicorn
import argparse
from fastapi import FastAPI

from schemas import SegmentationRequest
from schemas import SegmentationResponse
from schemas import Article


parser = argparse.ArgumentParser()
parser.add_argument("--port", "-p", type=int, default=8000)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--log_level", "-l", type=str, default="info") # Possible options are: critical, error, warning, info, debug, trace
parser.add_argument("--timeout", "-t", type=int, default=5) # Timeout in seconds
args = parser.parse_args()


app = FastAPI()


@app.post("/api/segment_article")
async def segment_article(request: SegmentationRequest) -> SegmentationResponse:
    pass


uvicorn.run(app, 
            port=args.port, 
            host=args.host, 
            log_level=args.log_level,
            timeout_keep_alive=args.timeout)

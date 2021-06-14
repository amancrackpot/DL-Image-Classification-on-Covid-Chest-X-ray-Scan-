from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.templating import Jinja2Templates
import uvicorn, aiohttp, asyncio
from io import BytesIO, StringIO
from fastai.vision.all import *
import requests
import base64
import pdb

export_file_name = 'export.pkl'
classes = ['Normal', 'Covid', 'Viral Pneumonia']
path = Path(__file__).parent

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

templates = Jinja2Templates(directory='src/templates')
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='src/static'))


async def setup_learner():
#     await download_file(export_file_url, path/'models'/export_file_name)
    defaults.device = torch.device('cpu')
    learn = load_learner(path/'saved'/export_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

	
def model_predict(img_b):
    img = PILImage.create(BytesIO(img_b))
    label, _, outputs = learn.predict(img)
    formatted_outputs = [str(i)+'%' for i in np.round(outputs.numpy()*100,2)]
    pred_probs = (classes, formatted_outputs)

    img_bytes  = img.to_bytes_format()
    img_data = base64.b64encode(img_bytes).decode()

    result = {"class":label, "probs":pred_probs, "image":img_data}
    return result
   


@app.route('/upload', methods=["POST"])
async def upload(request):
    data = await request.form()
    img_b = await (data["file"].read())
    result = model_predict(img_b)

    return templates.TemplateResponse('result.html', {'request' : request, 'result' : result})
	
@app.route("/classify-url", methods=["GET","POST"])
async def classify_url(request):
    img_b = await get_bytes(request.url)
	
    results = model_predict(img_b)
    return templates.TemplateResponse('result.html', {'request' : request, 'result' : result})
    
@app.route("/")
def form(request):
    return templates.TemplateResponse('index.html', {'request' : request})

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app = app, host="0.0.0.0", port=8080)

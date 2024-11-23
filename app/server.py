from app.chains.classification import get_classification_chain
from app.chains.clustering import get_clusters_chain
from app.chains.summarization import get_summarization_chain
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, path="/summary", runnable=get_summarization_chain())
add_routes(app, path="/classify", runnable=get_classification_chain())
add_routes(app, path="/clustering", runnable=get_clusters_chain())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

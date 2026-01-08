from fastapi import FastAPI

from app.routes import router
from core.model import OvisModelServer, QwenModelServer, set_models


def create_app() -> FastAPI:
    app = FastAPI(title="Ovis2 Serving", version="0.1.0")
    app.include_router(router)

    @app.on_event("startup")
    def _load_model() -> None:
        ovis_model = OvisModelServer.from_env()
        qwen_model = QwenModelServer.from_env()
        set_models(ovis_model, qwen_model)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8666)

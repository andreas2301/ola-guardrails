from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ola_gateway_shared.transport import caller_cn_dependency

from .engine import GuardrailsEngine


class _CheckInputRequest(BaseModel):
    prompt: str = Field(max_length=8192)


class _CheckOutputRequest(BaseModel):
    response: str = Field(max_length=8192)


def create_app(
    engine: GuardrailsEngine,
    allowed_cns: frozenset[str] = frozenset(),
) -> FastAPI:
    app = FastAPI()

    app.state.allowed_cns = allowed_cns
    app.state.caller_cn_dep = caller_cn_dependency
    caller_dep = Depends(app.state.caller_cn_dep)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/check/input")
    async def check_input(req: _CheckInputRequest, caller_cn: str = caller_dep) -> dict:
        try:
            allowed, reason = await run_in_threadpool(engine.check_input, req.prompt)
        except Exception:
            # Fail closed: a rail-backend outage must never allow the request.
            raise HTTPException(status_code=500, detail="input rail check failed") from None

        return {"allowed": allowed, "reason": reason}

    @app.post("/check/output")
    async def check_output(req: _CheckOutputRequest, caller_cn: str = caller_dep) -> dict:
        try:
            allowed, reason = await run_in_threadpool(engine.check_output, req.response)
        except NotImplementedError:
            raise HTTPException(status_code=501, detail="output rail not implemented") from None
        except Exception:
            # Fail closed: never echo the raw response on a backend error.
            raise HTTPException(status_code=500, detail="output rail check failed") from None

        return {"allowed": allowed, "reason": reason}

    return app

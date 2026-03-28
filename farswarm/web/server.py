"""Farswarm web API server."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="Farswarm",
    description="Brain-encoded swarm social simulation engine",
    version="0.1.0",
)


class RunRequest(BaseModel):
    """Request to run a simulation."""

    stimulus_path: str
    n_agents: int = 1000
    n_rounds: int = 168
    encoder: str = "mock"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None


class SimulationStatus(BaseModel):
    """Status of a running or completed simulation."""

    simulation_id: str
    status: str
    progress: float = 0.0


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "Farswarm", "version": "0.1.0"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/encoders")
async def list_encoders() -> dict[str, list[str]]:
    from farswarm.encoders.registry import encoder_registry

    return {"encoders": encoder_registry.list_encoders()}


@app.post("/run")
async def run_simulation(request: RunRequest) -> dict:
    """Run a full simulation pipeline."""
    from farswarm.config import FarswarmConfig
    from farswarm.core.llm import OpenAICompatibleBackend
    from farswarm.core.pipeline import FarswarmPipeline

    path = Path(request.stimulus_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Stimulus not found: {path}")

    config = FarswarmConfig(
        encoder_name=request.encoder,
        llm_model=request.llm_model,
        llm_base_url=request.llm_base_url,
    )
    llm = OpenAICompatibleBackend(
        model=request.llm_model,
        base_url=request.llm_base_url,
    )
    pipeline = FarswarmPipeline(config=config, llm=llm)
    result = await pipeline.run(
        path,
        n_agents=request.n_agents,
        n_rounds=request.n_rounds,
    )

    report = result["analysis"]["report"]
    return {
        "simulation_id": result["simulation_id"],
        "report": report.to_dict(),
        "signals": result["analysis"]["signals"].to_dict(),
    }


@app.post("/upload")
async def upload_stimulus(file: UploadFile) -> dict[str, str]:
    """Upload a stimulus file for processing."""
    upload_dir = Path("./uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "stimulus"
    dest = upload_dir / filename
    content = await file.read()
    dest.write_bytes(content)

    return {"path": str(dest), "filename": filename}


def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Farswarm web server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)

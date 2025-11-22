# backend/app.py
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from registry import ModelRegistry

app = FastAPI(title="Unified Inference API (local)")
REG = ModelRegistry("models.json")


class PredictIn(BaseModel):
    model: str = "multihead"
    models: Optional[List[str]] = None  # <— NEW
    text: Optional[str] = None
    texts: Optional[List[str]] = None


class PredictOut(BaseModel):
    model: str
    outputs: List[Dict[str, Any]]


class PredictMultiOut(BaseModel):
    results: Dict[str, List[Dict[str, Any]]]  # model_name -> outputs


@app.get("/healthz")
def health():
    return {"status": "ok", "models": REG.list_models()}


@app.get("/models")
def models():
    return {"available": REG.list_models()}


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    texts = inp.texts if inp.texts else ([inp.text] if inp.text else [])
    if not texts:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'.")
    try:
        mdl = REG.get(inp.model)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    outputs = mdl.predict(texts)
    return {"model": inp.model, "outputs": outputs}


@app.post("/predict_multi", response_model=PredictMultiOut)  # <— NEW
def predict_multi(inp: PredictIn):
    texts = inp.texts if inp.texts else ([inp.text] if inp.text else [])
    if not texts:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'.")

    if not inp.models:
        raise HTTPException(status_code=400, detail="Provide 'models': [...]")

    results: Dict[str, List[Dict[str, Any]]] = {}
    for name in inp.models:
        try:
            mdl = REG.get(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        results[name] = mdl.predict(texts)

    return {"results": results}


@app.post("/predict_multi_audio", response_model=PredictMultiOut)
async def predict_multi_audio(
    file: UploadFile = File(...),
    models: List[str] = Query(..., description="List of model names to run"),
):
    if not models:
        raise HTTPException(status_code=400, detail="Provide at least one model.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    # Save uploaded audio to a temporary file
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    except Exception:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    results: Dict[str, List[Dict[str, Any]]] = {}

    try:
        for name in models:
            try:
                mdl = REG.get(name)
            except KeyError as e:
                raise HTTPException(status_code=404, detail=str(e))

            # Our InferenceModel.predict expects a list of inputs.
            outputs = mdl.predict([tmp_path])  # one audio clip -> list with 1 dict
            results[name] = outputs

    finally:
        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return {"results": results}

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from starlette.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
import tempfile, shutil, os
import time
import base64
from utils.tajweed_audio_segmenter import TajweedAudioSegmenter
from utils.tajweed_utils import (
    process_audio_with_tajweed,
    visualize_tajweed_results,
    load_rule_trees
)
import torch
import nemo.collections.asr as nemo_asr
from services.tajweed_session_service import TajweedSessionService

router = APIRouter(
    prefix="/tajweed",
    tags=["Tajweed"],
)

# Load ASR model and rule trees once
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
asr_model.to(device)
rule_trees = load_rule_trees()
visualization_cache = {}
session_service = TajweedSessionService()

@router.post("/process/")
async def tajweed_process(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(
            status_code=400, detail="Only WAV or MP3 files are supported."
        )
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        result = process_audio_with_tajweed(temp_file_path, asr_model, rule_trees)
        if not result:
            raise HTTPException(status_code=500, detail="Tajweed processing failed.")
        # Generate the visualization and base64
        buffer = visualize_tajweed_results(result)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        file_id = f"tajweed_{int(time.time())}_{os.path.basename(temp_file_path)}"
        visualization_cache[file_id] = {"buffer": buffer, "base64": plot_base64, "timestamp": time.time()}
        # TajweedAudioSegmenter analysis
        
        segmenter = TajweedAudioSegmenter(audio_file_path=temp_file_path, tajweed_output=result)
        analysis = segmenter.analyze_tajweed()
        # Save session using service
        user_id = str(request.headers.get("user_id", "anonymous"))
        session_id = file_id  # or generate a UUID if needed
        transcription = result.get("transcription", "")
        print(f"Transcription: {transcription}")
        # Prepare dict for DB/service in the requested format using the actual result structure
        tajweed_data = {
            "surah_idx" : result.get('surah_idx', 1),
            "ayah_idx" : result.get('ayah_idx', 1),
            'transcription': result.get('transcription', ''),
            'transcription_is_correct' : result.get('transcription_is_correct' , False),
            'words': [
                {
                    'word': word_info.get('word'),
                    "word_is_correct": word_info.get('word_is_correct', False),
                    'start_idx': word_info.get('start_idx'),
                    'end_idx': word_info.get('end_idx'),
                    'tajweed_rules': [
                        {
                            'rule': rule.get('rule'),
                            'start_idx': rule.get('start_idx'),
                            'end_idx': rule.get('end_idx'),
                            'text': rule.get('text'),
                            'tajweed_is_correct': rule.get('spans_words')
                        } for rule in word_info.get('tajweed_rules', [])
                    ]
                } for word_info in result.get('words', [])
            ]
        }
        await session_service.create_session(
            user_id="2",  # Replace with actual user ID from request
            session_id=session_id,
            tajweed_data=tajweed_data,
        )
        response = {
            "analyze_tajweed": analysis,
            "result": result,
            "file_id": file_id,
            "visualization_base64": f"data:image/png;base64,{plot_base64}"
        }
        
        return JSONResponse(content=jsonable_encoder(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.get("/visualization/")
async def tajweed_visualization(request: Request, file_id: str):
    if file_id not in visualization_cache:
        raise HTTPException(status_code=404, detail="Visualization not found.")
    viz_data = visualization_cache[file_id]
    if time.time() - viz_data["timestamp"] > 3600:
        del visualization_cache[file_id]
        raise HTTPException(status_code=410, detail="Visualization expired.")
    viz_data["buffer"].seek(0)
    return StreamingResponse(viz_data["buffer"], media_type="image/png")

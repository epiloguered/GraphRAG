#!/usr/bin/env python3
"""
Simple but Complete Youtu-GraphRAG Backend
Integrates real GraphRAG functionality with a simple interface
"""

import os
import re
import sys
import json
import asyncio
import glob
import shutil
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from schemas_api.request_models import AskQuestionRequest, CompareQuestionRequest
from schemas_api.response_models import CompareQuestionResponse, StrategyRunResponse
from utils.logger import logger

# Import document parser
try:
    from utils.document_parser import get_parser
    DOCUMENT_PARSER_AVAILABLE = True
except ImportError as e:
    DOCUMENT_PARSER_AVAILABLE = False
    logger.warning(f"Document parser not available: {e}")

# Try to import GraphRAG components
try:
    from models.constructor import kt_gen as constructor
    from models.retriever import agentic_decomposer as decomposer, enhanced_kt_retriever as retriever
    from config import get_config, ConfigManager
    from services.qa_orchestrator import QAOrchestrator
    GRAPHRAG_AVAILABLE = True
    logger.info("✅ GraphRAG components loaded successfully")
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    logger.error(f"⚠️  GraphRAG components not available: {e}")

app = FastAPI(title="Youtu-GraphRAG Unified Interface", version="1.0.0")

# Mount static files (assets directory)
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
# Mount frontend directory for frontend assets
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
active_connections: Dict[str, WebSocket] = {}
config = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

# Request/Response models
class FileUploadResponse(BaseModel):
    success: bool
    message: str
    dataset_name: Optional[str] = None
    files_count: Optional[int] = None

class GraphConstructionRequest(BaseModel):
    dataset_name: str
    
class GraphConstructionResponse(BaseModel):
    success: bool
    message: str
    graph_data: Optional[Dict] = None

def ensure_demo_schema_exists() -> str:
    """Ensure default demo schema exists and return its path."""
    os.makedirs("schemas", exist_ok=True)
    schema_path = "schemas/demo.json"
    if not os.path.exists(schema_path):
        demo_schema = {
            "Nodes": [
                "person", "location", "organization", "event", "object",
                "concept", "time_period", "creative_work", "biological_entity", "natural_phenomenon"
            ],
            "Relations": [
                "is_a", "part_of", "located_in", "created_by", "used_by", "participates_in",
                "related_to", "belongs_to", "influences", "precedes", "arrives_in", "comparable_to"
            ],
            "Attributes": [
                "name", "date", "size", "type", "description", "status",
                "quantity", "value", "position", "duration", "time"
            ]
        }
        with open(schema_path, 'w') as f:
            json.dump(demo_schema, f, indent=2)
    return schema_path

def get_schema_path_for_dataset(dataset_name: str) -> str:
    """Return dataset-specific schema if present; otherwise fallback to demo schema."""
    if dataset_name and dataset_name != "demo":
        ds_schema = f"schemas/{dataset_name}.json"
        if os.path.exists(ds_schema):
            return ds_schema
    return ensure_demo_schema_exists()


qa_orchestrator = QAOrchestrator(manager, get_schema_path_for_dataset) if GRAPHRAG_AVAILABLE else None

async def send_progress_update(client_id: str, stage: str, progress: int, message: str):
    """Send progress update via WebSocket"""
    await manager.send_message({
        "type": "progress",
        "stage": stage,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }, client_id)

# -------- Encoding detection helpers --------
def _detect_encoding_from_bytes(data: bytes) -> Optional[str]:
    """Detect encoding using chardet if available; return lower-cased encoding name or None."""
    try:
        import chardet  # type: ignore
        result = chardet.detect(data) or {}
        enc = result.get("encoding")
        if enc:
            return enc.lower()
    except Exception:
        pass
    return None

def decode_bytes_with_detection(data: bytes) -> str:
    """Decode bytes to string with encoding detection and robust fallbacks.
    Order: detected -> utf-8/utf-8-sig -> common Chinese encodings -> utf-16 variants -> latin-1 -> replace.
    """
    candidates = []
    detected = _detect_encoding_from_bytes(data)
    if detected:
        candidates.append(detected)
    candidates.extend([
        "utf-8", "utf-8-sig", "gb18030", "gbk", "big5",
        "utf-16", "utf-16le", "utf-16be", "latin-1"
    ])
    # De-duplicate while preserving order
    tried = set()
    for enc in candidates:
        if enc in tried or not enc:
            continue
        tried.add(enc)
        try:
            return data.decode(enc)
        except Exception:
            continue
    # Last resort
    return data.decode("utf-8", errors="replace")

async def clear_cache_files(dataset_name: str):
    """Clear all cache files for a dataset before graph construction"""
    try:
        # Clear FAISS cache files
        faiss_cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(faiss_cache_dir):
            shutil.rmtree(faiss_cache_dir)
            logger.info(f"Cleared FAISS cache directory: {faiss_cache_dir}")
        
        # Clear output chunks
        chunk_file = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            logger.info(f"Cleared chunk file: {chunk_file}")
        
        # Clear output graphs
        graph_file = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_file):
            os.remove(graph_file)
            logger.info(f"Cleared graph file: {graph_file}")
        
        # Clear any other cache files with dataset name pattern
        cache_patterns = [
            f"output/logs/{dataset_name}_*.log",
            f"output/chunks/{dataset_name}_*",
            f"output/graphs/{dataset_name}_*"
        ]
        
        for pattern in cache_patterns:
            for file_path in glob.glob(pattern):
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleared cache file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Cleared cache directory: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clear {file_path}: {e}")
        
        logger.info(f"Cache cleanup completed for dataset: {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error clearing cache files for {dataset_name}: {e}")
        # Don't raise exception, just log the error

# Serve frontend HTML
@app.get("/")
async def read_root():
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Youtu-GraphRAG Unified Interface is running!", "status": "ok"}

@app.get("/api/status")
async def get_status():
    return {
        "message": "Youtu-GraphRAG Unified Interface is running!", 
        "status": "ok",
        "graphrag_available": GRAPHRAG_AVAILABLE
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_files(files: List[UploadFile] = File(...), client_id: str = "default"):
    """Upload files and prepare for graph construction"""
    try:
        # Generate dataset name based on file count
        if len(files) == 1:
            # Single file: use its name
            main_file = files[0]
            original_name = os.path.splitext(main_file.filename)[0]
            # Clean filename to be filesystem-safe
            dataset_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            dataset_name = dataset_name.replace(' ', '_')
        else:
            # Multiple files: create a descriptive name with date
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            dataset_name = f"{len(files)}files_{date_str}"
        
        # Add counter if dataset already exists
        base_name = dataset_name
        counter = 1
        while os.path.exists(f"data/uploaded/{dataset_name}"):
            dataset_name = f"{base_name}_{counter}"
            counter += 1
            
        upload_dir = f"data/uploaded/{dataset_name}"
        os.makedirs(upload_dir, exist_ok=True)
        
        await send_progress_update(client_id, "upload", 10, "Starting file upload...")
        
        # Process uploaded files
        corpus_data = []
        skipped_files: List[str] = []
        processed_count = 0
        allowed_extensions = {".txt", ".md", ".json", ".pdf", ".docx", ".doc"}
        
        # Initialize document parser if needed
        doc_parser = None
        if DOCUMENT_PARSER_AVAILABLE:
            doc_parser = get_parser()
        
        for i, file in enumerate(files):
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                content_bytes = await file.read()
                buffer.write(content_bytes)
            
            # Process file content using encoding detection
            filename_lower = (file.filename or "").lower()
            ext = os.path.splitext(filename_lower)[1]
            if ext not in allowed_extensions:
                # Skip unsupported file types to avoid processing binary files as text
                logger.warning(f"Skipping unsupported file type: {file.filename}")
                skipped_files.append(file.filename)
                progress = 10 + (i + 1) * 80 // len(files)
                await send_progress_update(client_id, "upload", progress, f"Skipped unsupported file: {file.filename}")
                continue
            
            # Handle PDF and DOCX/DOC files with document parser
            if ext in ['.pdf', '.docx', '.doc']:
                if not doc_parser:
                    logger.warning(f"Document parser not available, skipping {file.filename}")
                    skipped_files.append(file.filename)
                    progress = 10 + (i + 1) * 80 // len(files)
                    await send_progress_update(client_id, "upload", progress, f"Skipped {file.filename} (parser unavailable)")
                    continue
                
                try:
                    text = doc_parser.parse_file(file_path, ext)
                    if text and text.strip():
                        corpus_data.append({
                            "title": file.filename,
                            "text": text
                        })
                        processed_count += 1
                        await send_progress_update(client_id, "upload", 10 + (i + 1) * 80 // len(files), f"Parsed {file.filename}")
                    else:
                        logger.warning(f"No text extracted from {file.filename}")
                        skipped_files.append(file.filename)
                        await send_progress_update(client_id, "upload", 10 + (i + 1) * 80 // len(files), f"No text in {file.filename}")
                except Exception as e:
                    logger.error(f"Error parsing {file.filename}: {e}")
                    skipped_files.append(file.filename)
                    await send_progress_update(client_id, "upload", 10 + (i + 1) * 80 // len(files), f"Failed to parse {file.filename}")
                continue
            
            # Treat plain text formats explicitly (.txt and .md)
            if filename_lower.endswith(('.txt', '.md')):
                text = decode_bytes_with_detection(content_bytes)
                corpus_data.append({
                    "title": file.filename,
                    "text": text
                })
                processed_count += 1
            elif filename_lower.endswith('.json'):
                try:
                    json_text = decode_bytes_with_detection(content_bytes)
                    data_obj = json.loads(json_text)
                    if isinstance(data_obj, list):
                        corpus_data.extend(data_obj)
                    else:
                        corpus_data.append(data_obj)
                    processed_count += 1
                except Exception:
                    # If JSON parsing fails, treat as text
                    text = decode_bytes_with_detection(content_bytes)
                    corpus_data.append({
                        "title": file.filename,
                        "text": text
                    })
            
            progress = 10 + (i + 1) * 80 // len(files)
            await send_progress_update(client_id, "upload", progress, f"Processed {file.filename}")
        
        # Ensure at least one valid file processed
        if processed_count == 0:
            msg = "No supported files were uploaded. Allowed: .txt, .md, .json, .pdf, .docx, .doc"
            if skipped_files:
                msg += f"; skipped: {', '.join(skipped_files)}"
            await send_progress_update(client_id, "upload", 0, msg)
            raise HTTPException(status_code=400, detail=msg)
        
        # Save corpus data
        corpus_path = f"{upload_dir}/corpus.json"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(corpus_data, f, ensure_ascii=False, indent=2)
        
        # Create dataset configuration
        await create_dataset_config()
        
        await send_progress_update(client_id, "upload", 100, "Upload completed successfully!")
        
        msg_ok = "Files uploaded successfully"
        if skipped_files:
            msg_ok += f"; skipped unsupported: {', '.join(skipped_files)}"
        return FileUploadResponse(
            success=True,
            message=msg_ok,
            dataset_name=dataset_name,
            files_count=processed_count
        )
    
    except Exception as e:
        await send_progress_update(client_id, "upload", 0, f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def create_dataset_config():
    """Create dataset configuration"""
    # Ensure default demo schema exists
    ensure_demo_schema_exists()

@app.post("/api/construct-graph", response_model=GraphConstructionResponse)
async def construct_graph(request: GraphConstructionRequest, client_id: str = "default"):
    """Construct knowledge graph from uploaded data"""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")
        dataset_name = request.dataset_name
        
        await send_progress_update(client_id, "construction", 2, "Cleaning old cache files...")
        
        # Clear all cache files before construction
        await clear_cache_files(dataset_name)
        
        await send_progress_update(client_id, "construction", 5, "Initializing graph builder...")
        
        # Get dataset paths
        corpus_path = f"data/uploaded/{dataset_name}/corpus.json" 
        # Choose schema: dataset-specific or default demo
        schema_path = get_schema_path_for_dataset(dataset_name)
        
        if not os.path.exists(corpus_path):
            # Try demo dataset
            corpus_path = "data/demo/demo_corpus.json"
        
        if not os.path.exists(corpus_path):
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        await send_progress_update(client_id, "construction", 10, "Loading configuration and corpus...")
        
        # Initialize config
        global config
        if config is None:
            config = get_config("config/base_config.yaml")
        
        # Initialize KTBuilder
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )
        
        await send_progress_update(client_id, "construction", 20, "Starting entity-relation extraction...")
        
        # Build knowledge graph
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Run graph construction without simulated progress updates
        knowledge_graph = await loop.run_in_executor(None, build_graph_sync)
        
        await send_progress_update(client_id, "construction", 95, "Preparing visualization data...")
        # Load constructed graph for visualization
        graph_path = f"output/graphs/{dataset_name}_new.json"
        graph_vis_data = await prepare_graph_visualization(graph_path)
        
        await send_progress_update(client_id, "construction", 100, "Graph construction completed!")
        # Notify completion via WebSocket
        try:
            await manager.send_message({
                "type": "complete",
                "stage": "construction",
                "message": "Graph construction completed!",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send completion message: {_e}")
        
        return GraphConstructionResponse(
            success=True,
            message="Knowledge graph constructed successfully",
            graph_data=graph_vis_data
        )
    
    except Exception as e:
        await send_progress_update(client_id, "construction", 0, f"Construction failed: {str(e)}")
        try:
            await manager.send_message({
                "type": "error",
                "stage": "construction",
                "message": f"Construction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send error message: {_e}")
        raise HTTPException(status_code=500, detail=str(e))


async def prepare_graph_visualization(graph_path: str) -> Dict:
    """Prepare graph data for visualization"""
    try:
        if os.path.exists(graph_path):
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}
        
        # Handle different graph data formats
        if isinstance(graph_data, list):
            # GraphRAG format: list of relationships
            return convert_graphrag_format(graph_data)
        elif isinstance(graph_data, dict) and "nodes" in graph_data:
            # Standard format: {nodes: [], edges: []}
            return convert_standard_format(graph_data)
        else:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}
    
    except Exception as e:
        logger.error(f"Error preparing visualization: {e}")
        return {"nodes": [], "links": [], "categories": [], "stats": {}}

def convert_graphrag_format(graph_data: List) -> Dict:
    """Convert GraphRAG relationship list to ECharts format"""
    nodes_dict = {}
    links = []
    
    # Extract nodes and relationships from the list
    for item in graph_data:
        if not isinstance(item, dict):
            continue
            
        start_node = item.get("start_node", {})
        end_node = item.get("end_node", {})
        relation = item.get("relation", "related_to")
        
        # Process start node
        start_id = ""
        end_id = ""
        if start_node:
            start_id = start_node.get("properties", {}).get("name", "")
            if start_id and start_id not in nodes_dict:
                nodes_dict[start_id] = {
                    "id": start_id,
                    "name": start_id[:30],
                    "category": start_node.get("properties", {}).get("schema_type", start_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": start_node.get("properties", {})
                }
        
        # Process end node
        if end_node:
            end_id = end_node.get("properties", {}).get("name", "")
            if end_id and end_id not in nodes_dict:
                nodes_dict[end_id] = {
                    "id": end_id,
                    "name": end_id[:30],
                    "category": end_node.get("properties", {}).get("schema_type", end_node.get("label", "entity")),
                    "symbolSize": 25,
                    "properties": end_node.get("properties", {})
                }
        
        # Add relationship
        if start_id and end_id:
            links.append({
                "source": start_id,
                "target": end_id,
                "name": relation,
                "value": 1
            })
    
    # Create categories
    categories_set = set()
    for node in nodes_dict.values():
        categories_set.add(node["category"])
    
    categories = []
    for i, cat_name in enumerate(categories_set):
        categories.append({
            "name": cat_name,
            "itemStyle": {
                "color": f"hsl({i * 360 / len(categories_set)}, 70%, 60%)"
            }
        })
    
    nodes = list(nodes_dict.values())
    
    return {
        "nodes": nodes[:500],  # Limit for better visual effects​​
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(links),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000])
        }
    }

def convert_standard_format(graph_data: Dict) -> Dict:
    """Convert standard {nodes: [], edges: []} format to ECharts format"""
    nodes = []
    links = []
    categories = []
    
    # Extract unique categories
    node_types = set()
    for node in graph_data.get("nodes", []):
        node_type = node.get("type", "entity")
        node_types.add(node_type)
    
    for i, node_type in enumerate(node_types):
        categories.append({
            "name": node_type,
            "itemStyle": {
                "color": f"hsl({i * 360 / len(node_types)}, 70%, 60%)"
            }
        })
    
    # Process nodes
    for node in graph_data.get("nodes", []):
        nodes.append({
            "id": node.get("id", ""),
            "name": node.get("name", node.get("id", ""))[:30],
            "category": node.get("type", "entity"),
            "value": len(node.get("attributes", [])),
            "symbolSize": min(max(len(node.get("attributes", [])) * 3 + 15, 15), 40),
            "attributes": node.get("attributes", [])
        })
    
    # Process edges
    for edge in graph_data.get("edges", []):
        links.append({
            "source": edge.get("source", ""),
            "target": edge.get("target", ""),
            "name": edge.get("relation", "related_to"),
            "value": edge.get("weight", 1)
        })
    
    return {
        "nodes": nodes[:500],  # Limit for performance
        "links": links[:1000],
        "categories": categories,
        "stats": {
            "total_nodes": len(graph_data.get("nodes", [])),
            "total_edges": len(graph_data.get("edges", [])),
            "displayed_nodes": len(nodes[:500]),
            "displayed_edges": len(links[:1000])
        }
    }

@app.get("/api/retrieval-modes")
async def get_retrieval_modes():
    """Return the retrieval modes used by the strategy selector UI."""
    if qa_orchestrator is None:
        return {
            "modes": [
                {"id": "youtu_default", "name": "Youtu Default", "description": "Decomposition + hybrid retrieval + IRCoT"},
                {"id": "chunk_only", "name": "Chunk Only", "description": "Dense chunk retrieval only."},
                {"id": "triple_only", "name": "Triple Only", "description": "FAISS triple and community retrieval."},
                {"id": "node_relation_only", "name": "Node + Relation", "description": "Node and relation retrieval only."},
            ]
        }
    return {"modes": qa_orchestrator.list_modes()}


@app.post("/api/ask-question", response_model=StrategyRunResponse)
async def ask_question(request: AskQuestionRequest, client_id: str = "default"):
    """Run one retrieval strategy through the shared QA orchestrator."""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")
        return await qa_orchestrator.run_single(request, client_id=client_id)
    except FileNotFoundError as exc:
        await send_progress_update(client_id, "retrieval", 0, str(exc))
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        await send_progress_update(client_id, "retrieval", 0, str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        await send_progress_update(client_id, "retrieval", 0, f"Question answering failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/compare-question", response_model=CompareQuestionResponse)
async def compare_question(request: CompareQuestionRequest, client_id: str = "default"):
    """Run multiple strategies against the same question for side-by-side comparison."""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")
        return await qa_orchestrator.run_compare(request, client_id=client_id)
    except FileNotFoundError as exc:
        await send_progress_update(client_id, "compare", 0, str(exc))
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        await send_progress_update(client_id, "compare", 0, str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        await send_progress_update(client_id, "compare", 0, f"Comparison failed: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/datasets")
async def get_datasets():
    """Get list of available datasets"""
    datasets = []
    
    # Check uploaded datasets
    upload_dir = "data/uploaded"
    if os.path.exists(upload_dir):
        for item in os.listdir(upload_dir):
            item_path = os.path.join(upload_dir, item)
            if os.path.isdir(item_path):
                corpus_path = os.path.join(item_path, "corpus.json")
                if os.path.exists(corpus_path):
                    graph_path = f"output/graphs/{item}_new.json"
                    status = "ready" if os.path.exists(graph_path) else "needs_construction"
                    has_custom_schema = os.path.exists(f"schemas/{item}.json")
                    datasets.append({
                        "name": item,
                        "type": "uploaded",
                        "status": status,
                        "has_custom_schema": has_custom_schema
                    })
    
    # Add demo dataset
    demo_corpus = "data/demo/demo_corpus.json"
    if os.path.exists(demo_corpus):
        demo_graph = "output/graphs/demo_new.json"
        status = "ready" if os.path.exists(demo_graph) else "needs_construction"
        datasets.append({
            "name": "demo",
            "type": "demo", 
            "status": status,
            "has_custom_schema": False
        })
    
    return {"datasets": datasets}

@app.post("/api/datasets/{dataset_name}/schema")
async def upload_schema(dataset_name: str, schema_file: UploadFile = File(...)):
    """Upload a custom schema JSON for a dataset."""
    try:
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot upload schema for demo dataset")
        if not schema_file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="Schema file must be a .json file")

        content = await schema_file.read()
        try:
            schema_text = decode_bytes_with_detection(content)
            data = json.loads(schema_text)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Schema JSON must be an object")

        os.makedirs("schemas", exist_ok=True)
        save_path = f"schemas/{dataset_name}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {"success": True, "message": "Schema uploaded successfully", "dataset_name": dataset_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload schema: {str(e)}")

@app.delete("/api/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset and all its associated files"""
    try:
        if dataset_name == "demo":
            raise HTTPException(status_code=400, detail="Cannot delete demo dataset")
        
        deleted_files = []
        
        # Delete dataset directory
        dataset_dir = f"data/uploaded/{dataset_name}"
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
            deleted_files.append(dataset_dir)
        
        # Delete graph file
        graph_path = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_path):
            os.remove(graph_path)
            deleted_files.append(graph_path)
        
        # Delete schema file (if dataset-specific)
        schema_path = f"schemas/{dataset_name}.json"
        if os.path.exists(schema_path):
            os.remove(schema_path)
            deleted_files.append(schema_path)
        
        # Delete cache files
        cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            deleted_files.append(cache_dir)
        
        # Delete chunk files
        chunk_file = f"output/chunks/{dataset_name}.txt"
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
            deleted_files.append(chunk_file)
        
        return {
            "success": True,
            "message": f"Dataset '{dataset_name}' deleted successfully",
            "deleted_files": deleted_files
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

@app.post("/api/datasets/{dataset_name}/reconstruct")
async def reconstruct_dataset(dataset_name: str, client_id: str = "default"):
    """Reconstruct graph for an existing dataset"""
    try:
        if not GRAPHRAG_AVAILABLE:
            raise HTTPException(status_code=503, detail="GraphRAG components not available. Please install or configure them.")
        # Check if dataset exists
        corpus_path = f"data/uploaded/{dataset_name}/corpus.json"
        if not os.path.exists(corpus_path):
            if dataset_name == "demo":
                corpus_path = "data/demo/demo_corpus.json"
            else:
                raise HTTPException(status_code=404, detail="Dataset not found")
        
        await send_progress_update(client_id, "reconstruction", 5, "Starting reconstruction...")
        
        # Delete existing graph file
        graph_path = f"output/graphs/{dataset_name}_new.json"
        if os.path.exists(graph_path):
            os.remove(graph_path)
            await send_progress_update(client_id, "reconstruction", 15, "Old graph file deleted...")
        
        # Delete existing cache files
        cache_dir = f"retriever/faiss_cache_new/{dataset_name}"
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            await send_progress_update(client_id, "reconstruction", 25, "Cache files cleared...")
        
        await send_progress_update(client_id, "reconstruction", 35, "Reinitializing graph builder...")
        
        # Initialize config
        global config
        if config is None:
            config = get_config("config/base_config.yaml")
        
        # Choose schema: dataset-specific or default demo
        schema_path = get_schema_path_for_dataset(dataset_name)
        
        # Initialize KTBuilder
        builder = constructor.KTBuilder(
            dataset_name,
            schema_path,
            mode=config.construction.mode,
            config=config
        )
        
        await send_progress_update(client_id, "reconstruction", 50, "Rebuilding knowledge graph...")
        
        # Build knowledge graph
        def build_graph_sync():
            return builder.build_knowledge_graph(corpus_path)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Run graph reconstruction without simulated progress updates
        knowledge_graph = await loop.run_in_executor(None, build_graph_sync)
        
        await send_progress_update(client_id, "reconstruction", 100, "Graph reconstruction completed!")
        # Notify completion via WebSocket
        try:
            await manager.send_message({
                "type": "complete",
                "stage": "reconstruction",
                "message": "Graph reconstruction completed!",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send completion message: {_e}")
        
        return {
            "success": True,
            "message": "Dataset reconstructed successfully",
            "dataset_name": dataset_name
        }
    
    except Exception as e:
        await send_progress_update(client_id, "reconstruction", 0, f"Reconstruction failed: {str(e)}")
        try:
            await manager.send_message({
                "type": "error",
                "stage": "reconstruction",
                "message": f"Reconstruction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }, client_id)
        except Exception as _e:
            logger.warning(f"Failed to send error message: {_e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/{dataset_name}")
async def get_graph_data(dataset_name: str):
    """Get graph visualization data"""
    graph_path = f"output/graphs/{dataset_name}_new.json"
    
    if not os.path.exists(graph_path):
        # Return demo data
        return {
            "nodes": [
                {"id": "node1", "name": "Example Entity 1", "category": "person", "value": 5, "symbolSize": 25},
                {"id": "node2", "name": "Example Entity 2", "category": "location", "value": 3, "symbolSize": 20},
            ],
            "links": [
                {"source": "node1", "target": "node2", "name": "located_in", "value": 1}
            ],
            "categories": [
                {"name": "person", "itemStyle": {"color": "#ff6b6b"}},
                {"name": "location", "itemStyle": {"color": "#4ecdc4"}},
            ],
            "stats": {"total_nodes": 2, "total_edges": 1, "displayed_nodes": 2, "displayed_edges": 1}
        }
    
    return await prepare_graph_visualization(graph_path)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    os.makedirs("data/uploaded", exist_ok=True)
    os.makedirs("output/graphs", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)
    os.makedirs("output/experiments", exist_ok=True)
    os.makedirs("schemas", exist_ok=True)
    
    logger.info("🚀 Youtu-GraphRAG Unified Interface initialized")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

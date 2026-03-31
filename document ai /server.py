"""
M&A Due Diligence Tool - Backend (FastAPI) v3.2.1
Fixed filename matching and missing PDF handling
"""

import os
import json
import traceback
import glob
import sys
import asyncio
import uuid
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from functools import lru_cache

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import litellm
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TOOL_DIR = Path(__file__).parent.resolve()
DOCS_DIR = TOOL_DIR / "docs"
TREES_DIR = TOOL_DIR / "trees"
METADATA_DIR = TOOL_DIR / "metadata"

sys.path.insert(0, str(TOOL_DIR))
try:
    from pageindex.page_index import page_index
    PAGEINDEX_AVAILABLE = True
except ImportError as e:
    PAGEINDEX_AVAILABLE = False
    logger.warning(f"PageIndex not found: {e}")

load_dotenv(TOOL_DIR / ".env")

app = FastAPI(title="M&A Due Diligence Tool", version="3.2.1", docs_url=None, redoc_url=None)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET","POST"], allow_headers=["Authorization","Content-Type"])

# Security headers
from starlette.middleware.base import BaseHTTPMiddleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response
app.add_middleware(SecurityHeadersMiddleware)

MODEL = os.environ.get("MODEL", "gemini/gemini-2.5-flash")
API_KEY = os.environ.get("API_KEY", "")
security = HTTPBearer(auto_error=False)
jobs: Dict[str, dict] = {}

class LoadRequest(BaseModel):
    name: str

class AnalyzeRequest(BaseModel):
    doc_name: str = Field(...)
    node: Dict[str, Any] = Field(...)

class SearchRequest(BaseModel):
    query: str = Field(...)
    doc_name: Optional[str] = None
    tree: List[Any] = Field(default_factory=list)

class GapsRequest(BaseModel):
    doc_name: str
    checklist: Optional[List[str]] = None

def ensure_dirs():
    DOCS_DIR.mkdir(exist_ok=True)
    TREES_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)

def find_pdf_file(doc_name: str) -> Optional[Path]:
    """Find PDF file with case-insensitive matching."""
    # Direct match first
    for ext in ['.pdf', '.PDF']:
        p = DOCS_DIR / f"{doc_name}{ext}"
        if p.exists():
            return p
    
    # Case insensitive search
    doc_name_lower = doc_name.lower()
    for f in DOCS_DIR.glob("*.pdf"):
        if f.stem.lower() == doc_name_lower:
            return f
    return None

def is_scanned_pdf(pdf_path: Path) -> bool:
    """Check if PDF appears to be image-based."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        text_chars = 0
        for i, page in enumerate(doc[:5]):
            text = page.get_text()
            text_chars += len(text.strip())
        doc.close()
        return text_chars < 100
    except Exception:
        return False

def extract_with_ocr(pdf_path: Path) -> Dict[int, str]:
    """Fallback OCR extraction."""
    try:
        import fitz
        from PIL import Image
        import pytesseract
        
        logger.info(f"OCR extraction for {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        pages = {}
        
        for i, page in enumerate(doc):
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            pages[i + 1] = text
            
        doc.close()
        return pages
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return {}

@lru_cache(maxsize=10)
def get_cached_doc_pages(doc_name: str):
    return _extract_pdf_pages_internal(doc_name)

def _extract_pdf_pages_internal(doc_name: str):
    """Extract text from PDF with multiple fallback methods."""
    pdf_path = find_pdf_file(doc_name)
    
    if not pdf_path:
        raise FileNotFoundError(f"PDF file for '{doc_name}' not found in docs folder")

    # Try standard extraction
    pages = {}
    extraction_method = "standard"
    
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        for i, page in enumerate(doc):
            pages[i + 1] = page.get_text()
        doc.close()
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")
    
    # Check if we got meaningful text
    total_text = "".join(pages.values()).strip()
    if len(total_text) < 500 and len(pages) > 2:
        logger.warning("Low text, trying OCR...")
        ocr_pages = extract_with_ocr(pdf_path)
        if ocr_pages:
            pages = ocr_pages
            extraction_method = "ocr"
    
    # Fallback to pypdf
    if not pages:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                pages[i + 1] = page.extract_text() or ""
            extraction_method = "pypdf"
        except Exception as e:
            logger.error(f"All extraction methods failed: {e}")
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    if not pages or not any(p.strip() for p in pages.values()):
        raise HTTPException(status_code=400, detail="PDF contains no extractable text")
    
    # Save metadata
    try:
        metadata_path = METADATA_DIR / f"{doc_name}.json"
        meta = {"extraction_method": extraction_method, "page_count": len(pages)}
        with open(metadata_path, 'w') as f:
            json.dump(meta, f)
    except Exception as e:
        logger.error(f"Metadata error: {e}")
    
    return pages

def get_doc_pages(doc_name: str):
    try:
        return get_cached_doc_pages(doc_name)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Error getting doc pages: {e}")
        raise HTTPException(status_code=500, detail="Could not read document")

def get_section_text(doc_name: str, start_page: int, end_page: int, max_chars: int = 30000):
    pages = get_doc_pages(doc_name)
    text = ""
    for p in range(start_page, end_page + 1):
        if p in pages:
            text += f"\n--- Page {p} ---\n{pages[p]}"
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[...truncated...]"
            break
    return text

def get_full_doc_text(doc_name: str, max_chars: int = 60000):
    pages = get_doc_pages(doc_name)
    text_parts = []
    for p in sorted(pages.keys()):
        text_parts.append(f"\n--- Page {p} ---\n{pages[p]}\n")
        if sum(len(x) for x in text_parts) > max_chars:
            break
    text = "".join(text_parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated...]"
    return text

def create_basic_structure(pdf_path: Path):
    """Create chunked structure when TOC parsing fails."""
    try:
        pages = get_doc_pages(pdf_path.stem)
        total_pages = len(pages)
    except Exception:
        total_pages = 10
    
    structure = []
    chunk_size = 10
    for i in range(0, total_pages, chunk_size):
        start_page = i + 1
        end_page = min(i + chunk_size, total_pages)
        page_text = pages.get(start_page, "")[:200].replace('\n', ' ').strip()
        title = page_text[:60] + "..." if len(page_text) > 60 else page_text
        if not title or len(title) < 5:
            title = f"Pages {start_page}-{end_page}"
        
        structure.append({
            "node_id": f"chunk_{len(structure) + 1}",
            "title": title,
            "summary": f"Document section covering pages {start_page} to {end_page}",
            "start_index": start_page,
            "end_index": end_page,
            "nodes": []
        })
    return structure

def convert_tree(nodes, depth=0):
    result = []
    for i, node in enumerate(nodes):
        title = node.get("title", "Untitled") if isinstance(node, dict) else str(node)
        c = {
            "id": str(i + 1),
            "title": title,
            "summary": node.get("summary", "") if isinstance(node, dict) else "",
            "start_page": node.get("start_index") if isinstance(node, dict) else None,
            "end_page": node.get("end_index") if isinstance(node, dict) else None,
            "depth": depth,
            "children": [],
        }
        if c["start_page"] and c["end_page"]:
            c["summary"] = (c["summary"] or "") + f" (pp. {c['start_page']}-{c['end_page']})"
        if isinstance(node, dict) and "nodes" in node and node["nodes"]:
            c["children"] = convert_tree(node["nodes"], depth + 1)
        result.append(c)
    return result

def count_nodes(tree):
    c = 0
    for n in tree:
        c += 1
        if n.get("children"):
            c += count_nodes(n["children"])
    return c

def call_llm(context_text: str, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 4096):
    if system_prompt is None:
        system_prompt = """You are an expert analyst. Provide clear, natural language answers."""
    
    if not context_text or not context_text.strip():
        raise ValueError("No document content available")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Document Context:\n{context_text[:35000]}\n\nUser Request: {prompt}"}
    ]
    
    try:
        resp = litellm.completion(model=MODEL, messages=messages, max_tokens=max_tokens, temperature=0.3)
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=503, detail="AI service temporarily unavailable")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not API_KEY:
        return True
    if not credentials or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

DEFAULT_CHECKLIST = [
    "Corporate governance and board structure",
    "Financial statements and audit reports",
    "Outstanding debt and liabilities",
    "Pending or threatened litigation",
    "Regulatory compliance and licenses",
    "Intellectual property and patents",
    "Employment agreements and key personnel",
    "Material contracts and customer agreements",
    "Change of control provisions",
    "Environmental liabilities",
    "Tax compliance and outstanding obligations",
    "Insurance coverage",
    "Related party transactions",
    "Data privacy and cybersecurity",
    "Real estate and leases",
]

@app.get("/api/health")
def health():
    ensure_dirs()
    return {
        "status": "ok", 
        "model": MODEL, 
        "version": "3.2.1", 
        "pageindex_available": PAGEINDEX_AVAILABLE
    }

@app.get("/api/documents")
def list_documents():
    """Only list documents that have both structure AND pdf files."""
    ensure_dirs()
    docs = []
    
    for tree_file in sorted(glob.glob(str(TREES_DIR / "*_structure.json"))):
        name = os.path.basename(tree_file).replace("_structure.json", "")
        
        # CRITICAL: Check if PDF actually exists
        if not find_pdf_file(name):
            logger.warning(f"Skipping {name}: structure exists but PDF missing")
            continue
        
        metadata_path = METADATA_DIR / f"{name}.json"
        page_count = 0
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    page_count = metadata.get("page_count", 0)
            except Exception:
                pass
        
        try:
            with open(tree_file) as f:
                raw = json.load(f)
            tree = convert_tree(raw if isinstance(raw, list) else [raw])
            node_count = count_nodes(tree)
        except Exception:
            node_count = 0
        
        docs.append({
            "name": name, 
            "pages": page_count, 
            "nodes": node_count
        })
    
    return {"documents": docs}

@app.post("/api/load")
def load_document(req: LoadRequest):
    """Load document only if PDF exists."""
    try:
        # CRITICAL: Verify PDF exists before loading structure
        pdf_path = find_pdf_file(req.name)
        if not pdf_path:
            raise HTTPException(
                status_code=404, 
                detail=f"PDF file for '{req.name}' not found. Please upload the document again."
            )
        
        # Try to load structure
        tree_path = TREES_DIR / f"{req.name}_structure.json"
        
        if tree_path.exists():
            with open(tree_path) as f:
                raw = json.load(f)
            tree = convert_tree(raw if isinstance(raw, list) else [raw])
        else:
            # Create basic structure on the fly
            logger.info(f"No structure file for {req.name}, creating basic structure")
            structure = create_basic_structure(pdf_path)
            tree = convert_tree(structure)
        
        # Pre-validate we can read the PDF
        try:
            get_doc_pages(req.name)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Cannot read PDF {req.name}: {e}")
            raise HTTPException(status_code=400, detail="Cannot read PDF")
        
        return {"tree": tree, "name": req.name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        raise HTTPException(status_code=500, detail="Failed to load document")

@app.post("/api/analyze")
def analyze_section(req: AnalyzeRequest):
    try:
        if not req.doc_name or not req.node:
            raise HTTPException(status_code=400, detail="doc_name and node required")
        
        start = req.node.get("start_page") or 1
        end = req.node.get("end_page") or start + 5
        section_text = get_section_text(req.doc_name, start, end)
        
        if not section_text.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Empty content",
                    "answer": "⚠️ Could not extract text from this section. The PDF may be image-based without OCR.",
                    "section_title": req.node.get('title', ''),
                    "pages": f"{start}-{end}"
                }
            )

        system_prompt = """You are an expert CFA instructor analyzing a curriculum section."""
        
        prompt = f"""Analyze this section: {req.node.get('title', '')}
Pages: {start}-{end}

Provide a clear analysis with page references."""

        answer = call_llm(section_text, prompt, system_prompt=system_prompt, max_tokens=2048)
        
        return {
            "answer": answer,
            "section_title": req.node.get('title', ''),
            "pages": f"{start}-{end}",
            "doc_name": req.doc_name
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing section: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.post("/api/search")
def search(req: SearchRequest):
    try:
        if not req.query:
            raise HTTPException(status_code=400, detail="No query provided")
        
        if not req.doc_name:
            raise HTTPException(status_code=400, detail="No document selected")
        
        try:
            doc_text = get_full_doc_text(req.doc_name, max_chars=50000)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error reading document in search: {e}")
            raise HTTPException(status_code=400, detail="Error reading document")

        system_prompt = """You are an expert CFA instructor. Answer based on the document provided."""
        
        prompt = f"""Question: {req.query}

Provide a comprehensive answer with page citations."""

        answer = call_llm(doc_text, prompt, system_prompt=system_prompt, max_tokens=4096)
        
        answer = re.sub(r'^```\w*\n?', '', answer.strip())
        answer = re.sub(r'\n?```$', '', answer)
        
        confidence = "high"
        if any(x in answer.lower() for x in ["does not mention", "not found", "no information", "cannot determine"]):
            confidence = "low"
        
        return {
            "answer": answer,
            "question": req.query,
            "document": req.doc_name,
            "confidence": confidence
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

@app.post("/api/upload")
async def upload_and_build(file: UploadFile = File(...), auth: bool = Depends(verify_token)):
    try:
        ensure_dirs()
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        job_id = str(uuid.uuid4())
        
        # Security: sanitize filename — strip path components and non-safe chars
        raw_name = os.path.basename(file.filename)
        safe_basename = re.sub(r'[^a-zA-Z0-9_\-. ]', '', os.path.splitext(raw_name)[0])[:100]
        if not safe_basename:
            safe_basename = "upload"
        doc_name = safe_basename
        
        # Read with size limit
        content = await file.read(MAX_UPLOAD_SIZE + 1)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")
        
        # Security: validate PDF magic bytes
        if len(content) < 5 or not content[:5].startswith(b'%PDF-'):
            raise HTTPException(status_code=400, detail="File is not a valid PDF")
        
        pdf_path = DOCS_DIR / f"{doc_name}.pdf"
        
        # Handle duplicates
        counter = 1
        original_name = doc_name
        while pdf_path.exists():
            doc_name = f"{original_name}_{counter}"
            pdf_path = DOCS_DIR / f"{doc_name}.pdf"
            counter += 1
        
        with open(pdf_path, "wb") as f:
            f.write(content)
        
        file_size = len(content) / (1024 * 1024)
        logger.info(f"Saved {pdf_path.name} ({file_size:.1f} MB)")
        
        jobs[job_id] = {
            "id": job_id,
            "status": "processing",
            "filename": pdf_path.name,
            "doc_name": doc_name,
            "file_size_mb": round(file_size, 2),
            "created_at": datetime.now().isoformat(),
            "progress": "Starting..."
        }
        
        asyncio.create_task(process_pdf_background(job_id, pdf_path, doc_name))
        return {"job_id": job_id, "status": "processing", "doc_name": doc_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

async def process_pdf_background(job_id: str, pdf_path: Path, doc_name: str):
    try:
        loop = asyncio.get_event_loop()
        jobs[job_id]["progress"] = "Analyzing structure..."
        
        structure = None
        
        if PAGEINDEX_AVAILABLE:
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: page_index(str(pdf_path), model=MODEL)),
                    timeout=900
                )
                if isinstance(result, dict):
                    structure = result.get('structure', [])
                elif isinstance(result, list):
                    structure = result
            except Exception as e:
                logger.warning(f"PageIndex failed: {e}")
        
        if not structure:
            jobs[job_id]["progress"] = "Using fallback extraction..."
            structure = create_basic_structure(pdf_path)
        
        # Save structure
        dst_tree = TREES_DIR / f"{doc_name}_structure.json"
        with open(dst_tree, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        
        # Pre-extract pages for caching
        try:
            pages = get_doc_pages(doc_name)
            jobs[job_id]["progress"] = f"Indexed {len(pages)} pages"
        except Exception as e:
            jobs[job_id]["progress"] = f"Structure saved (text warning: {str(e)[:30]})"
        
        jobs[job_id].update({
            "status": "completed",
            "doc_name": doc_name,
            "progress": f"Complete - {len(structure)} sections"
        })
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        jobs[job_id].update({
            "status": "failed",
            "error": "Document processing failed"
        })

@app.post("/api/gaps")
def gap_analysis(req: GapsRequest, auth: bool = Depends(verify_token)):
    try:
        doc_text = get_full_doc_text(req.doc_name, max_chars=50000)
        if not doc_text:
            raise HTTPException(status_code=400, detail="Document contains no text")
        
        checklist = req.checklist or DEFAULT_CHECKLIST
        checklist_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(checklist))
        
        system_prompt = """You are conducting a document completeness review."""
        
        prompt = f"""Review this document against the checklist:
{checklist_text}

Provide a narrative report with page references."""

        answer = call_llm(doc_text, prompt, system_prompt=system_prompt, max_tokens=4096)
        
        return {
            "answer": answer,
            "doc_name": req.doc_name,
            "stats": {"total_items": len(checklist)}
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in gap analysis: {e}")
        raise HTTPException(status_code=500, detail="Gap analysis failed")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred"}
    )

# Security: Only serve the static/ subdirectory, NOT the entire project
STATIC_DIR = TOOL_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    ensure_dirs()
    port = int(os.environ.get("PORT", 8080))
    print(f"\n{'='*55}")
    print(f"  M&A Due Diligence Tool v3.2.1")
    print(f"  URL: http://localhost:{port}")
    print(f"  Model: {MODEL}")
    print(f"{'='*55}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
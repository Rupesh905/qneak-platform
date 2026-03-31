"""
Build script - indexes all PDFs in docs/ folder using PageIndex.

Usage:
  python build.py                    # Index all unindexed PDFs
  python build.py --force            # Re-index everything
  python build.py --file report.pdf  # Index one specific file
"""

import os
import sys
import glob
import json
from pathlib import Path
from datetime import datetime

# Add parent to path to import pageindex (same as server.py)
TOOL_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(TOOL_DIR))

try:
    from pageindex.page_index import page_index
    PAGEINDEX_AVAILABLE = True
except ImportError:
    PAGEINDEX_AVAILABLE = False
    print("❌ PageIndex not found. Please copy VectifyAI/PageIndex to ./pageindex/")
    sys.exit(1)

DOCS_DIR = TOOL_DIR / "docs"
TREES_DIR = TOOL_DIR / "trees"
METADATA_DIR = TOOL_DIR / "metadata"

MODEL = os.environ.get("MODEL", "gemini/gemini-2.5-flash")

def build_one(pdf_path: Path, force=False):
    """Build PageIndex tree for one PDF."""
    name = pdf_path.stem
    tree_dst = TREES_DIR / f"{name}_structure.json"
    metadata_dst = METADATA_DIR / f"{name}.json"

    if tree_dst.exists() and not force:
        print(f"  ⏭  {name} — already indexed (use --force to rebuild)")
        return True

    print(f"  📄 {name} — indexing...")
    
    try:
        # Call PageIndex directly (same as server.py)
        result = page_index(str(pdf_path), model=MODEL)
        
        # Extract structure
        if isinstance(result, dict):
            structure = result.get('structure', result)
        else:
            structure = result
        
        # Save tree
        with open(tree_dst, 'w', encoding='utf-8') as f:
            json.dump(structure, f, indent=2, ensure_ascii=False)
        
        # Save metadata (page count, etc.)
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            
            with open(metadata_dst, 'w') as f:
                json.dump({
                    "page_count": page_count,
                    "indexed_at": datetime.now().isoformat(),
                    "model": MODEL
                }, f)
        except Exception as e:
            print(f"     ⚠️  Could not extract metadata: {e}")
        
        print(f"  ✅ {name} — done ({len(structure)} top-level sections)")
        return True
        
    except Exception as e:
        print(f"  ❌ {name} — FAILED")
        print(f"     Error: {e}")
        return False

def main():
    DOCS_DIR.mkdir(exist_ok=True)
    TREES_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)

    force = "--force" in sys.argv
    specific = None
    if "--file" in sys.argv:
        idx = sys.argv.index("--file")
        if idx + 1 < len(sys.argv):
            specific = sys.argv[idx + 1]
            # Security: prevent path traversal
            if os.sep in specific or specific.startswith('.'):
                print(f"❌ Invalid filename: {specific}")
                sys.exit(1)

    pdfs = sorted(DOCS_DIR.glob("*.pdf"))

    if specific:
        pdf_path = DOCS_DIR / specific
        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            sys.exit(1)
        pdfs = [pdf_path]

    if not pdfs:
        print(f"\n  No PDFs found in {DOCS_DIR}/")
        print(f"  Drop your due diligence PDFs there and run again.\n")
        sys.exit(0)

    print(f"\n{'='*55}")
    print(f"  M&A Due Diligence - Batch Indexer")
    print(f"  Model: {MODEL}")
    print(f"  PDFs:  {len(pdfs)}")
    print(f"{'='*55}\n")

    success = 0
    failed = 0
    skipped = 0

    for pdf in pdfs:
        result = build_one(pdf, force=force)
        if result:
            if (TREES_DIR / f"{pdf.stem}_structure.json").exists():
                success += 1
            else:
                skipped += 1
        else:
            failed += 1

    print(f"\n  Done: {success} indexed, {skipped} skipped, {failed} failed")
    print(f"  Run 'python server.py' to start the web UI\n")

if __name__ == "__main__":
    main()
"""
End-to-end demo script for the Resume RAG Service.

Creates a synthetic resume PDF, ingests it through the full pipeline,
and performs a retrieval query — all without a running server.

Run:
    python demo.py
"""

import io
import os

import fitz

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DEBUG", "true")

from fastapi.testclient import TestClient
from app.main import create_app

DEMO_RESUME = """John Doe
john.doe@email.com | +1-555-0101 | LinkedIn: linkedin.com/in/johndoe

Summary
Senior Python Engineer with 7 years of experience building scalable backend
systems, REST APIs, and machine learning pipelines. Passionate about clean
architecture and developer tooling.

Experience
Senior Software Engineer — DataTech Inc (2021–Present)
- Built distributed data pipelines using Apache Kafka and Python.
- Designed FastAPI microservices serving 2M+ requests per day.
- Led a team of 5 engineers across 3 time zones.

Software Engineer — WebCorp (2018–2021)
- Developed REST APIs using Django REST Framework and PostgreSQL.
- Reduced API response time by 40% through query optimization.
- Implemented CI/CD pipelines using GitHub Actions and Docker.

Education
M.Tech Computer Science — IIT Bombay (2016–2018)
GPA: 9.2/10.0

B.Tech Computer Science — VJTI Mumbai (2012–2016)
GPA: 8.8/10.0

Skills
Python, FastAPI, Django, PostgreSQL, MongoDB, Redis, Docker, Kubernetes,
Apache Kafka, Machine Learning, scikit-learn, PyTorch, AWS, GitHub Actions

Projects
Resume RAG System
Semantic retrieval system for resume analysis using ChromaDB and
sentence-transformers. Processes PDFs end-to-end without LLM dependency.

Real-time Analytics Dashboard
Built a streaming analytics platform processing 500K events/hour using
Kafka, Python, and ClickHouse.

Certifications
AWS Certified Solutions Architect — Professional (2023)
Google Professional Data Engineer (2022)
Kubernetes Administrator (CKA) (2021)

Achievements
Best Engineer Award — DataTech Inc (2023)
Speaker — PyCon India 2022: "Building Production RAG Systems"
Open Source Contributor — FastAPI (15+ merged PRs)
"""


def make_pdf_bytes(text: str) -> bytes:
    """Create a PDF in memory from text."""
    doc = fitz.open()
    page = doc.new_page()
    # Split into lines and insert with proper spacing
    y = 72
    for line in text.split("\n"):
        page.insert_text((72, y), line, fontsize=9)
        y += 13
        if y > 750:  # new page if overflow
            page = doc.new_page()
            y = 72
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)
    return buf.getvalue()


def main():
    print("=" * 60)
    print("  Resume RAG Service — End-to-End Demo")
    print("=" * 60)

    app = create_app()
    client = TestClient(app)

    # ── Health Check ──────────────────────────────────────────
    print("\n[1] Health Check")
    r = client.get("/api/v1/health")
    h = r.json()
    print(f"    Status  : {h['status']}")
    print(f"    Version : {h['version']}")
    print(f"    Env     : {h['environment']}")

    # ── Upload Resume ─────────────────────────────────────────
    print("\n[2] Uploading Resume PDF...")
    pdf_bytes = make_pdf_bytes(DEMO_RESUME)
    print(f"    PDF size: {len(pdf_bytes) / 1024:.1f} KB")

    r = client.post(
        "/api/v1/resume/upload",
        files={"file": ("john_doe_resume.pdf", pdf_bytes, "application/pdf")},
    )
    assert r.status_code == 201, f"Upload failed: {r.text}"
    data = r.json()["data"]

    resume_id = data["resume_id"]
    print(f"    resume_id    : {resume_id}")
    print(f"    Chars extracted : {data['char_count']}")
    print(f"    Chunks stored   : {data['chunks_stored']}")

    # ── Retrieval Queries ─────────────────────────────────────
    queries = [
        "Python FastAPI backend development",
        "machine learning and data engineering",
        "AWS certifications and cloud",
        "education degree IIT",
        "open source contributions and awards",
    ]

    print("\n[3] Semantic Retrieval Queries")
    print("-" * 60)

    for query in queries:
        r = client.get(
            "/api/v1/retrieve",
            params={
                "query": query,
                "top_k": 2,
                "similarity_threshold": 0.3,
                "resume_id": resume_id,
            },
        )
        results = r.json()["data"]["results"]
        print(f"\n  Query : '{query}'")
        if results:
            for res in results:
                print(f"  Match : [{res['section'].upper()}] score={res['score']:.3f}")
                print(f"          {res['text'][:120].strip()}...")
        else:
            print("  Match : No results above threshold")

    print("\n" + "=" * 60)
    print("  Demo complete. All systems operational.")
    print("=" * 60)


if __name__ == "__main__":
    main()

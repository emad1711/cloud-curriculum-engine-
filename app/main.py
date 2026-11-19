from __future__ import annotations

import os
import json
import uuid
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


BASE = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE / "storage" / "uploads"
PROCESSED_DIR = BASE / "storage" / "processed"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="Cloud Curriculum Engine",
    version="0.1.0",
    description="تحويل الكتب الدراسية إلى دروس تفاعلية"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)


class GenerateRequest(BaseModel):
    document_id: str
    lesson_title: str | None = None
    grade: str = "الرابع الابتدائي"
    subject: str = "الرياضيات"
    page_start: int | None = None
    page_end: int | None = None
    curriculum: str = "السعودي"


def extract_pdf(path: Path) -> list[dict[str, Any]]:
    try:
        import fitz
    except ImportError:
        raise HTTPException(
            500,
            "PyMuPDF غير مثبت. ثبّت المتطلبات أولاً."
        )

    doc = fitz.open(path)
    pages = []

    for idx, page in enumerate(doc):
        text = page.get_text("text").strip()
        method = "embedded_text"

        if len(text) < 35:
            method = "ocr_required"

            try:
                import pytesseract
                from PIL import Image

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )

                img_path = PROCESSED_DIR / (
                    f"ocr_{path.stem}_{idx + 1}.png"
                )

                pix.save(img_path)

                text = pytesseract.image_to_string(
                    Image.open(img_path),
                    lang=os.getenv("OCR_LANG", "ara+eng"),
                    config="--psm 6"
                ).strip()

                method = "tesseract_ocr"

            except Exception:
                text = ""

        pages.append({
            "page": idx + 1,
            "text": text,
            "extraction_method": method,
            "needs_review": (
                method != "embedded_text" or len(text) < 80
            )
        })

    return pages


def split_sections(
    pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    sections = []

    current = {
        "title": "محتوى غير مصنف",
        "pages": [],
        "text": []
    }

    heading = re.compile(
        r"(?:الدرس\s*[٠-٩0-9\-ـ]+|"
        r"الفصل\s*[٠-٩0-9]+|"
        r"الوحدة\s*[٠-٩0-9]+|"
        r"القيمة المنزلية ضمن مئات الألوف)"
    )

    for page in pages:
        lines = [
            x.strip()
            for x in page["text"].splitlines()
            if x.strip()
        ]

        detected = next(
            (
                x for x in lines[:12]
                if heading.search(x)
            ),
            None
        )

        if detected and (current["pages"] or current["text"]):
            sections.append({
                **current,
                "text": "\n".join(current["text"])
            })

            current = {
                "title": detected[:140],
                "pages": [],
                "text": []
            }

        current["pages"].append(page["page"])

        if page["text"]:
            current["text"].append(page["text"])

    if current["pages"] or current["text"]:
        sections.append({
            **current,
            "text": "\n".join(current["text"])
        })

    return sections


def llm_generate(
    source_text: str,
    req: GenerateRequest,
    page_refs: list[int]
) -> dict[str, Any]:

    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash"
    ).strip()

    if not key:
        raise HTTPException(
            503,
            "Gemini غير متصل. أضف GEMINI_API_KEY إلى ملف .env."
        )

    system = """
أنت محرك تعليمي متخصص في تحويل المناهج السعودية
إلى دروس تفاعلية للأطفال.

المهمة:
تحويل صفحات المصدر المحددة إلى درس مبسط مناسب
لطالب الصف الرابع الابتدائي (9-10 سنوات).

قواعد إلزامية:

1. التزم بالمصدر المرفق ولا تخترع معلومات أو أهدافاً
   وتنسبها للكتاب.

2. لا تحذف المفاهيم أو القواعد أو الخطوات الأساسية
   اللازمة لفهم الدرس.

3. الأمثلة الإضافية مسموحة بشرط وسمها بوضوح بأنها
   أمثلة تدريبية مولدة وليست منقولة من الكتاب.

4. استخدم عربية سهلة مناسبة للطفل مع الحفاظ على
   المصطلحات الرياضية الصحيحة.

5. اجعل الأسئلة والإجابات صحيحة رياضياً واشرح
   سبب الإجابة.

6. إذا كان المصدر ناقصاً أو OCR غير واضح،
   اذكر ذلك في source_coverage_notes ولا تخمن.

7. أعد JSON صالحاً فقط دون Markdown أو نص خارجه.

المفاتيح المطلوبة:
title, learning_goals, short_explanation, key_terms,
worked_examples, interactive_activities, quiz,
summary, source_coverage_notes.

في quiz استخدم:
question, choices, correct_index, explanation.

في كل نشاط حدد:
type, question, options عند الحاجة,
correct_answer, feedback.
"""

    user = {
        "curriculum": req.curriculum,
        "grade": req.grade,
        "subject": req.subject,
        "lesson_title": (
            req.lesson_title or
            "استخرج عنوان الدرس من المصدر"
        ),
        "source_pages": page_refs,
        "source_text": source_text[:28000],
        "instruction": (
            "إذا كان النص غير واضح أو لا يكفي لتحديد "
            "مفهوم الدرس، لا تخمن. وضح النقص."
        )
    }

    url = (
        f"https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    payload = {
        "systemInstruction": {
            "parts": [{"text": system}]
        },
        "contents": [{
            "role": "user",
            "parts": [{
                "text": json.dumps(
                    user,
                    ensure_ascii=False
                )
            }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    try:
        retryable_statuses = {
            429, 500, 502, 503, 504
        }

        response = None

        for attempt in range(5):
            try:
                response = httpx.post(
                    url,
                    params={"key": key},
                    json=payload,
                    timeout=180
                )

            except httpx.RequestError:
                if attempt == 4:
                    raise

                delay = 2 ** attempt

                print(
                    f"Gemini connection error. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)
                continue

            if (
                response.status_code in retryable_statuses
                and attempt < 4
            ):
                delay = 2 ** attempt

                print(
                    f"Gemini temporary error: "
                    f"{response.status_code}. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)
                continue

            break

        if response is None:
            raise HTTPException(
                503,
                "Gemini is temporarily unavailable."
            )

        response.raise_for_status()

        body = response.json()

        candidates = body.get("candidates") or []

        if not candidates:
            raise ValueError(
                "Gemini لم يرجع محتوى."
            )

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        content = "".join(
            part.get("text", "")
            for part in parts
        )

        if not content.strip():
            raise ValueError(
                "استجابة Gemini فارغة."
            )

        result = json.loads(content)

        required = {
            "title",
            "learning_goals",
            "short_explanation",
            "quiz"
        }

        missing = required - result.keys()

        if missing:
            raise ValueError(
                "Gemini response missing fields: "
                + ", ".join(sorted(missing))
            )

        result["source_pages"] = page_refs
        result["curriculum"] = req.curriculum
        result["grade"] = req.grade
        result["subject"] = req.subject
        result["review_status"] = "needs_human_review"
        result["ai_provider"] = "google_gemini"

        return result

    except HTTPException:
        raise

    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300]

        detail = re.sub(
            r"key=[^&\s]+",
            "key=[REDACTED]",
            detail
        )

        status_code = e.response.status_code

        if status_code in (429, 503):
            raise HTTPException(
                status_code,
                f"Gemini is temporarily unavailable: {detail}"
            )

        raise HTTPException(
            502,
            f"Gemini API error ({status_code}): {detail}"
        )

    except Exception as e:
        raise HTTPException(
            502,
            f"فشل توليد الدرس باستخدام Gemini: {str(e)[:240]}"
        )


@app.get("/", response_class=HTMLResponse)
def home():
    return (
        BASE / "web" / "index.html"
    ).read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cloud-curriculum-engine",
        "llm_configured": bool(
            os.getenv("GEMINI_API_KEY")
        ),
        "ai_provider": "google_gemini",
        "model": os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )
    }


def process_book_file(
    doc_id: str,
    filename: str,
    curriculum: str,
    grade: str,
    subject: str
):
    dest = UPLOAD_DIR / f"{doc_id}.pdf"

    try:
        pages = extract_pdf(dest)

        record = {
            "document_id": doc_id,
            "filename": filename,
            "curriculum": curriculum,
            "grade": grade,
            "subject": subject,
            "status": "ready",
            "page_count": len(pages),
            "pages_with_text": sum(
                bool(x["text"]) for x in pages
            ),
            "pages": pages,
            "sections": split_sections(pages),
            "processed_at": time.time()
        }

        (
            PROCESSED_DIR / f"{doc_id}.json"
        ).write_text(
            json.dumps(
                record,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    except Exception as e:
        state = {
            "document_id": doc_id,
            "filename": filename,
            "status": "failed",
            "error": str(e)
        }

        (
            PROCESSED_DIR / f"{doc_id}.json"
        ).write_text(
            json.dumps(
                state,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )


@app.post("/api/books/upload")
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    curriculum: str = "السعودي",
    grade: str = "الرابع الابتدائي",
    subject: str = "الرياضيات"
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            400,
            "النسخة الحالية تستقبل PDF فقط"
        )

    doc_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{doc_id}.pdf"

    max_mb = int(
        os.getenv("MAX_UPLOAD_MB", "100")
    )

    data = await file.read(
        (max_mb * 1024 * 1024) + 1
    )

    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(
            413,
            f"الحد الأقصى لحجم الملف {max_mb}MB"
        )

    if not data.startswith(b"%PDF-"):
        raise HTTPException(
            400,
            "الملف المرفوع ليس PDF صالحاً"
        )

    dest.write_bytes(data)

    state = {
        "document_id": doc_id,
        "filename": file.filename,
        "curriculum": curriculum,
        "grade": grade,
        "subject": subject,
        "status": "processing"
    }

    (
        PROCESSED_DIR / f"{doc_id}.json"
    ).write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8"
    )

    threading.Thread(
        target=process_book_file,
        args=(
            doc_id,
            file.filename,
            curriculum,
            grade,
            subject
        ),
        daemon=True
    ).start()

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "status": "processing",
        "status_url": f"/api/books/{doc_id}/status",
        "message": (
            "تم استلام الكتاب. يجري استخراج "
            "الصفحات في الخلفية."
        )
    }


@app.get("/api/books/{document_id}/status")
def book_status(document_id: str):
    p = PROCESSED_DIR / f"{document_id}.json"

    if not p.exists():
        raise HTTPException(
            404,
            "الكتاب غير موجود"
        )

    data = json.loads(
        p.read_text(encoding="utf-8")
    )

    if data.get("status") == "ready":
        return {
            "document_id": document_id,
            "status": "ready",
            "page_count": data.get("page_count"),
            "pages_with_text": sum(
                bool(x["text"])
                for x in data.get("pages", [])
            ),
            "ocr_review_pages": [
                x["page"]
                for x in data.get("pages", [])
                if x.get("needs_review")
            ],
            "sections": [
                {
                    "title": x["title"],
                    "pages": x["pages"]
                }
                for x in data.get("sections", [])
            ]
        }

    return {
        "document_id": document_id,
        "status": data.get("status"),
        "error": data.get("error")
    }


@app.get("/api/books/{document_id}")
def get_book(document_id: str):
    p = PROCESSED_DIR / f"{document_id}.json"

    if not p.exists():
        raise HTTPException(
            404,
            "الكتاب غير موجود"
        )

    data = json.loads(
        p.read_text(encoding="utf-8")
    )

    return {
        k: v
        for k, v in data.items()
        if k != "pages"
    }


@app.post("/api/lessons/generate")
def generate_lesson(req: GenerateRequest):
    p = PROCESSED_DIR / f"{req.document_id}.json"

    if not p.exists():
        raise HTTPException(
            404,
            "الكتاب غير موجود"
        )

    book = json.loads(
        p.read_text(encoding="utf-8")
    )

    if book.get("status") != "ready":
        raise HTTPException(
            409,
            "الكتاب لم يكتمل تحليله بعد."
        )

    if (
        req.page_start is not None
        and req.page_end is not None
    ):
        if (
            req.page_start < 1
            or req.page_end < req.page_start
            or req.page_end > book.get("page_count", 0)
        ):
            raise HTTPException(
                400,
                "نطاق الصفحات غير صالح."
            )

        selected = [
            x for x in book["pages"]
            if req.page_start <= x["page"] <= req.page_end
        ]

        source_text = "\n".join(
            x["text"]
            for x in selected
            if x["text"]
        )

        refs = [
            x["page"]
            for x in selected
        ]

    else:
        candidates = [
            s for s in book["sections"]
            if req.lesson_title
            and req.lesson_title in s["title"]
        ]

        section = (
            candidates[0]
            if candidates
            else None
        )

        if not section:
            raise HTTPException(
                422,
                "لم يتم تحديد الدرس بدقة. أدخل عنوان الدرس "
                "أو حدد page_start و page_end."
            )

        source_text = section.get("text", "")
        refs = section["pages"]

    if len(source_text.strip()) < 100:
        raise HTTPException(
            422,
            "النص المستخرج من الصفحات غير كافٍ. "
            "تحقق من OCR قبل التوليد."
        )

    result = llm_generate(
        source_text,
        req,
        refs
    )

    lesson_id = str(uuid.uuid4())

    result["lesson_id"] = lesson_id
    result["document_id"] = req.document_id

    (
        PROCESSED_DIR / f"lesson_{lesson_id}.json"
    ).write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return result


@app.get("/api/books/{document_id}/lessons")
def list_book_lessons(document_id: str):
    book_path = PROCESSED_DIR / f"{document_id}.json"

    if not book_path.exists():
        raise HTTPException(
            404,
            "الكتاب غير موجود"
        )

    items = []

    for path in PROCESSED_DIR.glob("lesson_*.json"):
        try:
            item = json.loads(
                path.read_text(encoding="utf-8")
            )

            if item.get("document_id") == document_id:
                items.append({
                    "lesson_id": item.get("lesson_id"),
                    "title": item.get("title"),
                    "source_pages": item.get("source_pages", []),
                    "review_status": item.get("review_status")
                })

        except Exception:
            continue

    return {
        "document_id": document_id,
        "lessons": items
    }


@app.get(
    "/api/lessons/{lesson_id}/student",
    response_class=HTMLResponse
)
def student_lesson(lesson_id: str):
    p = PROCESSED_DIR / f"lesson_{lesson_id}.json"

    if not p.exists():
        raise HTTPException(
            404,
            "الدرس غير موجود"
        )

    return (
        BASE / "web" / "lesson.html"
    ).read_text(
        encoding="utf-8"
    ).replace(
        "__LESSON_ID__",
        lesson_id
    )


@app.get("/api/lessons/{lesson_id}")
def get_lesson(lesson_id: str):
    p = PROCESSED_DIR / f"lesson_{lesson_id}.json"

    if not p.exists():
        raise HTTPException(
            404,
            "الدرس غير موجود"
        )

    return json.loads(
        p.read_text(encoding="utf-8")
    )
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fitz
from fastapi.testclient import TestClient
from app.main import app, split_sections, extract_pdf, PROCESSED_DIR

client = TestClient(app)

def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['ai_provider'] == 'google_gemini'

def test_split_sections_keeps_page_references():
    pages = [
        {'page': 14, 'text': 'الدرس ١-١ القيمة المنزلية ضمن مئات الألوف\nشرح القيمة المنزلية'},
        {'page': 15, 'text': 'الصيغة اللفظية والصيغة التحليلية'}
    ]
    sections = split_sections(pages)
    assert sections
    assert 14 in sections[0]['pages']

def test_generate_requires_ready_book_and_explicit_lesson():
    response = client.post('/api/lessons/generate', json={'document_id':'missing-book'})
    assert response.status_code == 404

def test_pdf_extraction_returns_source_pages(tmp_path):
    pdf_path = tmp_path / 'mini.pdf'
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72,72), 'Sample math lesson 12345')
    doc.save(pdf_path)
    doc.close()
    pages = extract_pdf(pdf_path)
    assert len(pages) == 1
    assert pages[0]['page'] == 1
    assert 'Sample' in pages[0]['text']

def test_book_lessons_endpoint_requires_existing_book():
    response = client.get('/api/books/missing-book/lessons')
    assert response.status_code == 404

def test_student_lesson_page_requires_existing_lesson():
    response = client.get('/api/lessons/missing-lesson/student')
    assert response.status_code == 404

def test_upload_rejects_non_pdf():
    response = client.post('/api/books/upload', files={'file': ('bad.pdf', b'not a pdf', 'application/pdf')})
    assert response.status_code == 400

import fitz  # PyMuPDF

def load_pdf_and_split(filepath, max_length=300):
    doc = fitz.open(filepath)
    full_text = ""

    for page in doc:
        text = page.get_text()
        full_text += text + "\n"

    # 简单按段落切分（可按需优化）
    raw_passages = [p.strip() for p in full_text.split("\n") if len(p.strip()) > 30]
    passages = []

    buf = ""
    for p in raw_passages:
        if len(buf) + len(p) < max_length:
            buf += " " + p
        else:
            passages.append(buf.strip())
            buf = p
    if buf:
        passages.append(buf.strip())

    return passages

import json
import re
import requests
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import pymupdf4llm
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def extract_bullets_from_table(content: str) -> list[str]:
    prompt = f"""
        You are a data extractor.

        Extract the table below into a list of JSON objects with keys:
        - ticker (e.g., "MSFT")
        - purchase_date (YYYY-MM-DD format)
        - price (float)
        - shares (int)

        Output a JSON array only. No extra text.

        Table:
        {content}
            """
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma:2b-instruct",
            "prompt": prompt,
            "stream": False
        })
        raw = res.json()["response"].strip()

        match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        json_str = match.group(1) if match else raw

        rows = json.loads(json_str)          
        return [
            f"- {r['ticker']}: {r['shares']} shares @ {r['price']} (bought {r['purchase_date']})"
            for r in rows
        ]
    except Exception as e:
        print("LLM extraction failed:", e)
        return []


def extract_markdown(pdf_path: str) -> str:
    return pymupdf4llm.to_markdown(pdf_path)

def extract_all_markdown_tables(md: str) -> list[str]:
    tables = []
    current = []
    for line in md.splitlines():
        if "|" in line:
            current.append(line)
        elif current and line.strip() == "":
            tables.append("\n".join(current))
            current = []
        elif current:
            tables.append("\n".join(current))
            current = []
    if current:
        tables.append("\n".join(current))
    return tables

def extract_image_chunks(pdf_path: str):
    return partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        languages=["eng"]
    )

def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def analyze_chart_image_openai(image_b64):
    vision_model = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        ("user", [
            {"type": "text", "text": "Extract tickers, purchase date, price and shares from this chart as JSON."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ])
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | vision_model

    res = chain.invoke({})
    raw = res.content
    match = re.search(r'```json\n(.*?)```', raw, re.DOTALL)
    try:
        return json.loads(match.group(1)) if match else []
    except:
        return []

def ingest(pdf_path, output_path="summaries.json"):
    all_summaries = []

    # Extract markdown tables
    md = extract_markdown(pdf_path)
    md_tables = extract_all_markdown_tables(md)
    print(f"🔍 Markdown tables found: {len(md_tables)}")

    for md_table in md_tables:
        bullets = extract_bullets_from_table(md_table)
        if bullets:
            for b in bullets:
                all_summaries.append({
                    "source": "markdown",
                    "type": "purchase_entry",
                    "summary": b,        
                    "raw": md_table
                })

    # Extract structured text/tables/images
    chunks = extract_image_chunks(pdf_path)
    for chunk in chunks:
        if chunk.category == "Table":
            summary = extract_bullets_from_table(chunk.text)
            all_summaries.append({"source": "pdf", 
                                  "type": "table", 
                                  "summary": summary, 
                                  "raw": chunk.text})
        elif chunk.category in {"NarrativeText", "CompositeElement"}:
            all_summaries.append({"source": "pdf", 
                                  "type": "text", 
                                  "raw": chunk.text})

    # Extract chart images and analyze
    chart_images = get_images_base64(chunks)
    for img in chart_images:
        result = analyze_chart_image_openai(img)
        if result:
            all_summaries.append({"source": "image", "type": "chart", "extracted": result})

    Path(output_path).write_text(json.dumps(all_summaries, indent=2))
    print(f"✅ Saved summaries to {output_path}")

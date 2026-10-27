import json

INVOICE_CONFIG = {
    "system": """You are a precise data-extraction assistant.
Extract invoice data from the text the user provides.

Rules:
- Use only information present in the text. Never invent values.
- vendor is the company that ISSUED the invoice (the seller), not the client / "billed to".
- Dates must be formatted YYYY-MM-DD.
- invoice_number is the identifier only (e.g. "2291"), without the word "Invoice" or "#".
- line_items contain products or services (shipping/delivery charges may be a line item).
  Taxes and VAT are NOT line items: put the tax amount in "tax".
  Discounts are NOT line items: put the amount in "discount" as a positive number.
- unit_price is the price of ONE unit, not the line total.
- Include EVERY line item listed in the document. Never skip one.
- quantity is the number of units billed (the number before "x" or in the Qty column).
  Numbers like "200 pcs" inside a description are NOT the quantity.
- Numbers must be plain numbers with a dot as decimal separator and no currency symbols or
  thousands separators (e.g. "1.200,50 EUR" becomes 1200.50).
- currency is a 3-letter code (£ = GBP, € = EUR, $ = USD).
- Reply with JSON only.""",
    "examples": [
        (
            """INV-0045 | Northwind Cafe Supplies
12 Jan 2026
2 x Espresso beans 1kg @ 18.50 = 37.00
10 x Paper cups (pack) @ 2.40 = 24.00
Sales tax 8%: 4.88
Amount due: $65.88""",
            {
                "vendor": "Northwind Cafe Supplies",
                "invoice_number": "INV-0045",
                "date": "2026-01-12",
                "currency": "USD",
                "line_items": [
                    {"description": "Espresso beans 1kg", "quantity": 2, "unit_price": 18.50},
                    {"description": "Paper cups (pack)", "quantity": 10, "unit_price": 2.40},
                ],
                "discount": 0,
                "tax": 4.88,
                "total": 65.88,
            },
        ),
        (
            """Invoice No. 7710
Ridgeway Print Co.
Billed to: Alder & Co
Date: 3 Nov 2025
100 x Flyers A5 @ 0.35 = 35.00
5 x Posters A2 @ 12.00 = 60.00
Subtotal 95.00
Early-payment discount -9.50
VAT 20% 17.10
Total GBP 102.60""",
            {
                "vendor": "Ridgeway Print Co.",
                "invoice_number": "7710",
                "date": "2025-11-03",
                "currency": "GBP",
                "line_items": [
                    {"description": "Flyers A5", "quantity": 100, "unit_price": 0.35},
                    {"description": "Posters A2", "quantity": 5, "unit_price": 12.00},
                ],
                "discount": 9.50,
                "tax": 17.10,
                "total": 102.60,
            },
        ),
    ],
}

RESUME_CONFIG = {
    "system": """You are a precise data-extraction assistant.
Extract resume data from the text the user provides.

Rules:
- Use only information present in the text. Never invent values.
- Search the WHOLE text for the email and phone number: they can appear anywhere.
  Use null only if they truly do not exist in the text.
- Dates: YYYY-MM if the month is known, otherwise YYYY. Use "present" for current jobs
  ("now", "current", "ongoing", "until today").
- Include EVERY job and EVERY degree mentioned. Do not stop after the first one.
- Education year is the graduation (END) year as 4 digits: "2016-2018" becomes "2018".
- For self-employed or freelance work, use "Freelance" as the company.
- If no education is mentioned, return an empty list.
- skills: one short skill per list item (split "Docker / Kubernetes" into two items).
- Reply with JSON only.""",
    "examples": [
        (
            """MARIA KOWALSKA
Warsaw, Poland | maria.k@example.com | +48 600 123 456

EXPERIENCE
Data Analyst, Orbit Retail (Mar 2022 - present)
- Built weekly sales dashboards in Power BI
Junior Analyst, Delta Logistics (2020 - Feb 2022)

EDUCATION
BSc Statistics, University of Warsaw, 2020

SKILLS: Python, SQL, Power BI, Excel""",
            {
                "full_name": "Maria Kowalska",
                "email": "maria.k@example.com",
                "phone": "+48 600 123 456",
                "location": "Warsaw, Poland",
                "skills": ["Python", "SQL", "Power BI", "Excel"],
                "experience": [
                    {"company": "Orbit Retail", "job_title": "Data Analyst", "start_date": "2022-03", "end_date": "present"},
                    {"company": "Delta Logistics", "job_title": "Junior Analyst", "start_date": "2020", "end_date": "2022-02"},
                ],
                "education": [
                    {"institution": "University of Warsaw", "degree": "BSc Statistics", "year": "2020"}
                ],
            },
        ),
        (
            """Tomas Berg
Frontend Developer, Stockholm
Contact: tomas.berg@mailbox.example

Employment history
Nordlight AB - Frontend Developer - 2022 until today
Pixelhaus - Web Developer - Aug 2019 to Jan 2022

Education
KTH Royal Institute of Technology, BSc Software Engineering, 2016-2019

Skills
JavaScript, React, TypeScript, CSS""",
            {
                "full_name": "Tomas Berg",
                "email": "tomas.berg@mailbox.example",
                "phone": None,
                "location": "Stockholm",
                "skills": ["JavaScript", "React", "TypeScript", "CSS"],
                "experience": [
                    {"company": "Nordlight AB", "job_title": "Frontend Developer", "start_date": "2022", "end_date": "present"},
                    {"company": "Pixelhaus", "job_title": "Web Developer", "start_date": "2019-08", "end_date": "2022-01"},
                ],
                "education": [
                    {"institution": "KTH Royal Institute of Technology", "degree": "BSc Software Engineering", "year": "2019"}
                ],
            },
        ),
    ],
}

DOC_CONFIGS = {"invoice": INVOICE_CONFIG, "resume": RESUME_CONFIG}

USER_TEMPLATE = "Extract the data from this {doc_type} text:\n\n{document_text}"


def build_messages(doc_type: str, document_text: str) -> list[dict]:
    """Assemble: instructions, worked examples (few-shot), then the real document."""
    cfg = DOC_CONFIGS[doc_type]
    messages = [{"role": "system", "content": cfg["system"]}]
    for example_input, example_output in cfg["examples"]:
        messages.append({"role": "user", "content": USER_TEMPLATE.format(doc_type=doc_type, document_text=example_input)})
        messages.append({"role": "assistant", "content": json.dumps(example_output)})
    messages.append({"role": "user", "content": USER_TEMPLATE.format(doc_type=doc_type, document_text=document_text)})
    return messages
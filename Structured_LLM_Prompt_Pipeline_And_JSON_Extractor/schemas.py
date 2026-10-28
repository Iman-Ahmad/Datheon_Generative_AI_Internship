import re
from datetime import datetime
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PARTIAL_DATE_RE = re.compile(r"\d{4}(-(0[1-9]|1[0-2]))?|present")
NULL_WORDS = {"", "null", "none", "n/a", "na", "-", "not provided", "not specified"}


def null_if_placeholder(v):
    """Models sometimes write the TEXT 'null' instead of a real null. Treat those as missing."""
    if isinstance(v, str) and v.strip().lower() in NULL_WORDS:
        return None
    return v

def get_source(info: ValidationInfo) -> str | None:
    """The original document text, passed in by extractor.py so we can cross-check the model."""
    return (info.context or {}).get("source_text")


def squash(text: str) -> str:
    """Lowercase and remove all whitespace, so line breaks from PDFs don't break comparisons."""
    return re.sub(r"\s+", "", text).lower()


def numbers_in(text: str) -> set[float]:
    """Find every number in the text, understanding both 1,250.00 and 1.250,00 styles."""
    found = set()
    for token in re.findall(r"\d[\d.,]*\d|\d", text):
        m = re.fullmatch(r"(.*?)[.,](\d{1,2})", token)   # last separator + 1-2 digits = decimals
        if m:
            whole = re.sub(r"[.,]", "", m.group(1)) or "0"
            found.add(float(f"{whole}.{m.group(2)}"))
        else:
            found.add(float(re.sub(r"[.,]", "", token)))
    return found


def amount_in_source(value: float, source: str) -> bool:
    return any(abs(value - n) < 0.005 for n in numbers_in(source))


# ---------- Invoice ----------
class LineItem(BaseModel):
    description: str = Field(description="Product or service name only. Never taxes, VAT or discounts.")
    quantity: float = Field(description="Number of units BILLED (the number before 'x' or in a Qty column). NOT a pack size like '200 pcs' written inside the description.")
    unit_price: float = Field(description="Price of ONE unit")


class Invoice(BaseModel):
    vendor: str = Field(description="Company that ISSUED the invoice (the seller), not the client")
    invoice_number: str = Field(description="The identifier only, without words like 'Invoice' or '#'")
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    currency: str = Field(description="3-letter ISO currency code: EUR, USD, GBP ...")
    line_items: list[LineItem]
    discount: float = Field(description="Total discount as a POSITIVE number. 0 if none.")
    tax: float = Field(description="Total tax amount (VAT / sales tax). 0 if none.")
    total: float = Field(description="Final amount due, exactly as written in the document")

    @field_validator("invoice_number")
    @classmethod
    def clean_invoice_number(cls, v):
        cleaned = re.sub(r"^\s*invoice\s*(no\.?|number)?\s*[#:]*\s*", "", v, flags=re.IGNORECASE).strip()
        return cleaned or v

    @field_validator("date")
    @classmethod
    def check_date_format(cls, v):
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @field_validator("currency")
    @classmethod
    def check_currency(cls, v):
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", v):
            raise ValueError("currency must be a 3-letter code such as EUR, USD or GBP")
        return v

    @field_validator("discount")
    @classmethod
    def discount_positive(cls, v):
        return abs(v)   # "-52.10" and "52.10" both mean a 52.10 discount

    @model_validator(mode="after")
    def check_totals(self):
        subtotal = sum(i.quantity * i.unit_price for i in self.line_items)
        expected = subtotal - self.discount + self.tax
        if abs(expected - self.total) > 0.05:
            raise ValueError(
                f"Items ({subtotal:.2f}) - discount ({self.discount:.2f}) + tax ({self.tax:.2f}) "
                f"= {expected:.2f}, but total is {self.total:.2f}. "
                "The total written in the document is correct, so do NOT change total, tax or discount. "
                "Instead check line_items: include EVERY item listed, and make sure each quantity "
                "and unit_price match the document (quantity = units billed, not pieces per pack)."
            )
        return self

    @model_validator(mode="after")
    def check_against_source(self, info: ValidationInfo):
        source = get_source(info)
        if not source:
            return self
        if squash(self.invoice_number) not in squash(source):
            raise ValueError(f"invoice_number '{self.invoice_number}' does not appear in the text")
        # Grounding check: never accept an amount that is not written in the document
        for name, value in (("total", self.total), ("tax", self.tax), ("discount", self.discount)):
            if value > 0 and not amount_in_source(value, source):
                raise ValueError(
                    f"{name} {value:.2f} does not appear in the document. "
                    "Copy amounts exactly as written; never recalculate or invent them."
                )
        return self

# ---------- Resume ----------
class WorkExperience(BaseModel):
    company: str
    job_title: str
    start_date: str | None = Field(default=None, description="YYYY-MM, or YYYY if month unknown, or null")
    end_date: str | None = Field(default=None, description="YYYY-MM, or YYYY, or 'present' if current, or null")

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def placeholder_to_none(cls, v):
        return null_if_placeholder(v)

    @field_validator("start_date", "end_date")
    @classmethod
    def check_dates(cls, v):
        if v is not None and not PARTIAL_DATE_RE.fullmatch(v):
            raise ValueError(f"'{v}' must look like 2021, 2021-06 or 'present'")
        return v


class Education(BaseModel):
    institution: str
    degree: str
    year: str | None = Field(
        default=None,
        description="Graduation (END) year as 4 digits, e.g. '2018'. null if still studying or no year is written. Never a grade or language level.",
    )

    @field_validator("year", mode="before")
    @classmethod
    def clean_year(cls, v):
        v = null_if_placeholder(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, str):
            m = re.fullmatch(r"\s*((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})\s*", v)
            if m:
                return m.group(2)   # "2016-2018" becomes "2018"
        return v

    @field_validator("year")
    @classmethod
    def check_year(cls, v):
        if v is not None and not re.fullmatch(r"\d{4}", v):
            raise ValueError(
                f"year '{v}' is not a 4-digit year. Grades (like 5A or 75.09) and language levels "
                "(like B1) are NOT years. Use null when no graduation year is written."
            )
        return v


class Resume(BaseModel):
    full_name: str
    email: str | None = Field(default=None, description="Email address, or null ONLY if none exists anywhere in the text")
    phone: str | None = Field(default=None, description="Phone number, or null if not in the text")
    location: str | None = Field(default=None, description="City/country, or null if not in the text")
    skills: list[str] = Field(description="Individual skills, one short item each")
    experience: list[WorkExperience]
    education: list[Education]

    @field_validator("email", "phone", "location", mode="before")
    @classmethod
    def placeholder_to_none(cls, v):
        return null_if_placeholder(v)

    @field_validator("email")
    @classmethod
    def check_email(cls, v):
        if v is not None and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError(f"'{v}' is not a valid email address")
        return v

    @model_validator(mode="after")
    def check_against_source(self, info: ValidationInfo):
        source = get_source(info)
        if not source:
            return self
        squashed = squash(source)
        found = EMAIL_RE.findall(source)
        if self.email is None and found:
            raise ValueError(f"The text contains an email address ({found[0]}) but email is null")
        if self.email is not None and squash(self.email) not in squashed:
            raise ValueError(f"email '{self.email}' does not appear in the text")
        for job in self.experience:
            if job.company.lower() != "freelance" and squash(job.company) not in squashed:
                raise ValueError(
                    f"Company '{job.company}' does not appear in the text. List only jobs that are "
                    "actually written in the document. If it lists no jobs, return an empty list "
                    "for experience. Never copy jobs from the examples."
                )
        return self
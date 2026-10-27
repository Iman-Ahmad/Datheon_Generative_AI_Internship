import argparse
import re
import time
from pathlib import Path

from extractor import DEFAULT_MODEL, extract
from loaders import read_document

JOBS = [("invoice", "samples/invoices"), ("resume", "samples/resumes")]

parser = argparse.ArgumentParser(description="Run the extractor on every sample document.")
parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
args = parser.parse_args()

out_dir = Path("outputs") / re.sub(r"[^\w.-]", "_", args.model)
out_dir.mkdir(parents=True, exist_ok=True)
rows = []

for doc_type, folder in JOBS:
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() not in {".txt", ".pdf", ".docx"}:
            continue
        expect_fail = path.stem.startswith("broken")   # files meant to be rejected
        start = time.time()
        try:
            result = extract(read_document(str(path)), doc_type, model=args.model)
            (out_dir / f"{path.name}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
            status = "UNEXPECTED PASS" if expect_fail else "OK"
        except Exception as e:
            status = "REJECTED (expected)" if expect_fail else "FAIL"
        rows.append((path.name, status, time.time() - start))
        print(f"{status:20} {path}")

good = sum(1 for _, s, _ in rows if s in ("OK", "REJECTED (expected)"))
lines = [f"Model: {args.model}", f"Result: {good}/{len(rows)} as expected", ""]
lines += [f"| {name} | {status} | {secs:.0f}s |" for name, status, secs in rows]
summary = "\n".join(lines)
(out_dir / "summary.md").write_text(summary, encoding="utf-8")
print("\n" + summary)
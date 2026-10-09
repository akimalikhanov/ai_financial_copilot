"""Build the ingestion load-test fixture set (docs/notes/loadtest-readiness-audit.md §8, 1.2-1.3).

Three classes, each exercising a different path through docling_parser.parse():

    normal.pdf      a real filing, whole             -> parse_status "success"
    scanned.pdf     the same filing, rasterized      -> "success_scan_ocr"  (no text layer)
    brokenfont.pdf  synthetic, wrong ToUnicode CMap  -> "success_ocr_fallback"

The first two are *capacity* fixtures: they carry service times the model consumes, so their
length must be representative. brokenfont is a *path* fixture — a yes/no on whether the
wrong-ToUnicode route reaches the OCR converter — so its length carries no information and is
set only by text_quality._MIN_SAMPLE_CHARS, the 2,000-char floor below which assess() declines
to judge at all.

The last two both end in the forced-OCR converter but arrive by different routes — one is
caught by classify_page_content() reading the PDF's page objects, the other by assess()
scoring the extracted text. A fixture set with only one of them leaves half the OCR path
untested, which is why both are here.

`normal` needs a real document and cannot be synthesized or truncated: T11 measures service
time, and a generated PDF parses in seconds where a real filing with tables and figures takes
far longer. Point --source at any filing under data/corpus/pdfs/ — that corpus runs 11-1043
pages, median 128, so check the page count printed below against it before trusting a run.

Fixtures are written to infra/k8s/loadtest/fixtures/, which is gitignored and mounted into
the kind node (see kind-cluster.yaml extraMounts + loadtest/fixtures-pv.yaml). Real filings
must not be committed, and generated PDFs are reproducible from this script, so nothing here
belongs in git.

Usage:
    .venv/bin/python -m infra.loadtest.make_fixtures --source "data/corpus/pdfs/Some Co.pdf"
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUT = Path("infra/k8s/loadtest/fixtures")

# Codepoint offset between the glyph drawn and the character the ToUnicode CMap claims it is.
# Any nonzero value reproduces the bug; 3 matches the shift measured on the filing that
# prompted Phase 8 ("financial" extracted as "ILQDQFLDO").
_TOUNICODE_SHIFT = 3

# Enough prose per page that a few pages clear text_quality's _MIN_SAMPLE_CHARS (2,000) floor.
# Below it `assess` declines to judge and the fixture silently fails to trigger the fallback —
# which is exactly what the first draft of this fixture did at 3 pages / ~1,950 characters.
_PROSE = [
    "ACME CORPORATION ANNUAL REPORT 2024",
    "Total revenue for the year was 4,821 million, an increase of 12.4%",
    "over the prior year, driven by growth in the services segment and",
    "continued expansion of the subscription base. Operating margin",
    "improved to 18.2% from 16.7%, reflecting operating leverage and",
    "disciplined cost management across the whole of the organization.",
    "Cash and cash equivalents totaled 1,204 million at the year end,",
    "compared with 998 million in the prior year. The increase was",
    "primarily attributable to cash generated from operations of the",
    "group, partially offset by capital expenditures in the period.",
]


def _pdf_bytes(objects: list[bytes]) -> bytes:
    """Assemble numbered objects into a PDF with a correct xref table.

    Hand-built because pypdfium2 reads text objects but cannot create them, and the fixtures
    need exact control over the font dictionary anyway.
    """
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n".encode()
    )
    return bytes(out)


def _stream_obj(payload: bytes) -> bytes:
    return b"<< /Length " + str(len(payload)).encode() + b" >>\nstream\n" + payload + b"\nendstream"


def _tounicode_cmap() -> bytes:
    """A ToUnicode CMap that maps every drawn byte to byte + _TOUNICODE_SHIFT.

    This is what makes the fixture faithful rather than merely garbled-looking: the glyphs
    still render as correct English, so OCR reads real text back, while anything reading the
    text layer gets nonsense. Writing pre-shifted characters into the page instead would
    render nonsense too, and OCR would then "recover" the same nonsense — exercising the
    fallback's failure branch rather than its recovery.
    """
    entries = "".join(f"<{code:02X}> <{code + _TOUNICODE_SHIFT:04X}>\n" for code in range(32, 127))
    return f"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CMapName /BrokenSubset def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
{len(range(32, 127))} beginbfchar
{entries}endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end""".encode()


def build_brokenfont(pages: int = 8) -> bytes:
    """A text-bearing PDF whose text layer decodes to garbage."""
    cmap = _tounicode_cmap()
    font_num = 3 + pages * 2
    cmap_num = font_num + 1
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(pages))

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode(),
    ]
    for i in range(pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {4 + i * 2} 0 R "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> >>".encode()
        )
        y = 720
        ops = []
        for line in _PROSE:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            ops.append(f"BT /F1 12 Tf 55 {y} Td ({escaped}) Tj ET")
            y -= 28
        objects.append(_stream_obj("\n".join(ops).encode()))
    objects.append(
        f"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode {cmap_num} 0 R >>".encode()
    )
    objects.append(_stream_obj(cmap))
    return _pdf_bytes(objects)


def build_normal(source: Path, out: Path, pages: int = 0) -> Path:
    """A real filing with its text layer intact. `pages`=0 means the whole document.

    Whole by default, and the default is the point. Truncating this fixture was a measurement
    bug: the corpus runs 11-1043 pages (median 128), so the 10-page version was shorter than
    the *smallest* real filing and reported a service time ~13x below the median document's.
    The leading pages of an annual report are also the shareholder letter — prose — so the
    dense financial tables that TableFormer is slow on were missing entirely.

    Truncation belongs to build_scanned (rasterizing costs ~0.4MB/page), not here.
    """
    import pypdfium2 as pdfium

    src = pdfium.PdfDocument(str(source))
    take = len(src) if pages <= 0 else min(pages, len(src))
    dest = pdfium.PdfDocument.new()
    dest.import_pages(src, list(range(take)))
    dest.save(str(out))
    src.close()
    return out


def build_scanned(source: Path, out: Path, pages: int = 0, dpi: int = 200) -> Path:
    """The same filing rasterized: page images, no text layer at all. `pages`=0 means all.

    Whole by default, for the same reason build_normal is. This fixture carries the *worst
    case* service time — OCR ran at ~2.7 s/page against ~0.12 s/page for a text PDF, so a
    median 128-page scan is ~6 minutes of single-slot occupancy, which is the head-of-line
    block that decides whether ingestion needs admission control. Truncating it also sampled
    only the leading pages, which on an annual report are the shareholder letter: prose, with
    far less text per page to OCR than the financial tables further in.

    ~0.41 MB/page, so a 93-page filing lands near 38MB — inside both MAX_FILE_SIZE (100MB,
    documents.py) and nginx's client_max_body_size (200m). `pages` stays as an escape hatch
    for a quick smoke run, not as the default.

    Grayscale because that is what document scanners actually emit, and it roughly halves the
    fixture size.
    """
    import pypdfium2 as pdfium
    from PIL import Image

    src = pdfium.PdfDocument(str(source))
    count = len(src) if pages <= 0 else min(pages, len(src))
    # pdfium renders at 72 DPI * scale. The stub types `scale` as int, but it is a float at
    # runtime and must be — 200 DPI is a scale of 2.78, and rounding it would silently change
    # the fixture's resolution.
    scale = dpi / 72.0
    images: list[Image.Image] = [
        src[i].render(scale=scale).to_pil().convert("L")  # pyright: ignore[reportArgumentType]
        for i in range(count)
    ]
    images[0].save(out, save_all=True, append_images=images[1:], resolution=float(dpi), quality=80)
    src.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="real PDF to derive normal/scanned from"
    )
    parser.add_argument(
        "--pages", type=int, default=0, help="pages of --source for normal.pdf; 0 = all of it"
    )
    parser.add_argument(
        "--scan-pages",
        type=int,
        default=0,
        help="pages to rasterize for scanned.pdf; 0 = all of it (~0.4MB/page)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--only",
        choices=["normal", "scanned", "brokenfont"],
        help="build a single class instead of all three",
    )
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"no such file: {args.source}")
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = {args.only} if args.only else {"normal", "scanned", "brokenfont"}
    built: list[Path] = []

    if "normal" in wanted:
        built.append(build_normal(args.source, args.out / "normal.pdf", args.pages))
    if "scanned" in wanted:
        built.append(
            build_scanned(args.source, args.out / "scanned.pdf", args.scan_pages, args.dpi)
        )
    if "brokenfont" in wanted:
        path = args.out / "brokenfont.pdf"
        path.write_bytes(build_brokenfont())
        built.append(path)

    # Verify rather than trust: a fixture that does not trip its detector is worse than no
    # fixture, because the run still produces numbers — they just measure the wrong path.
    import pypdfium2 as pdfium

    from src.services.ingestion import text_quality

    expected = {"normal": "text", "scanned": "scanned", "brokenfont": "text"}
    print()
    for path in built:
        verdict, _ = text_quality.classify_page_content(path)
        size_mb = path.stat().st_size / (1024 * 1024)
        doc = pdfium.PdfDocument(str(path))
        n_pages = len(doc)
        doc.close()
        want = expected[path.stem]
        flag = "ok" if verdict == want else f"MISMATCH (wanted {want})"
        # Page count is printed because truncation is the failure this script already shipped
        # once: a 10-page normal.pdf measured a service time an order of magnitude too low and
        # nothing in the output said so.
        print(f"  {path.name:16} {n_pages:5d}p {size_mb:6.1f} MB  classify={verdict:8} {flag}")

    if "brokenfont" in wanted:
        # classify_page_content says "text" for this one by design — it has a text layer.
        # The garbled check is what must fire, and it needs enough characters to judge.
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(args.out / "brokenfont.pdf"))
        extracted = "".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
        garbled, ratio = text_quality.assess(extracted, threshold=0.02)
        print(
            f"  brokenfont       garbled={garbled} stopword_ratio={ratio:.5f} chars={len(extracted)}"
        )
        if not garbled:
            print("  WARNING: brokenfont will not trigger the OCR fallback")

    print(f"\nwrote {len(built)} fixture(s) to {args.out}")


if __name__ == "__main__":
    main()

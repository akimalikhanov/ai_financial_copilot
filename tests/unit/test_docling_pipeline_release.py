"""The per-page release hook (findings doc §2 Option A).

Real Docling model objects, not mocks: the whole fix is that it reaches into private page
attributes of a pinned docling, so a version bump that moves them must fail here rather than
silently restore the 13.1 MB/page slope that OOM-kills the worker at ~1000 pages.

The pipeline is not constructed — that loads layout and table models — so the hook is invoked
against a stand-in carrying just the pipeline_options it reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from docling.datamodel.base_models import (
    AssembledUnit,
    Cluster,
    FigureElement,
    Page,
    TextElement,
)
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.labels import DocItemLabel
from PIL import Image

from src.services.ingestion.docling_pipeline import PictureAwarePdfPipeline


class _Backend:
    def __init__(self) -> None:
        self.unloaded = False

    def unload(self) -> None:
        self.unloaded = True


@dataclass
class _Item:
    """Stands in for docling's ThreadedItem: the hook reads only .payload."""

    payload: Page | None


def _element(label: DocItemLabel, element_id: int, page_no: int) -> Any:
    cluster = Cluster(
        id=element_id,
        label=label,
        bbox=BoundingBox(l=0, t=0, r=10, b=10),
    )
    if label == DocItemLabel.PICTURE:
        return FigureElement(label=label, id=element_id, page_no=page_no, cluster=cluster)
    return TextElement(label=label, id=element_id, page_no=page_no, cluster=cluster, text="x")


def _page(*labels: DocItemLabel, assembled: bool = True) -> Page:
    page = Page(page_no=1)
    page._backend = _Backend()  # type: ignore[assignment]
    # The three renders a real page accumulates: preprocessing's unconditional scale-1.0 pass,
    # the crop scale (images_scale, mirrored into _default_image_scale), and the scale-2.0
    # render the table-structure stage takes.
    page._default_image_scale = 1.5
    page._image_cache = {
        1.0: Image.new("RGB", (4, 4)),
        1.5: Image.new("RGB", (6, 6)),
        2.0: Image.new("RGB", (8, 8)),
    }
    page.predictions.layout = object()  # type: ignore[assignment]
    if assembled:
        page.assembled = AssembledUnit(
            elements=[_element(label, i, 1) for i, label in enumerate(labels)]
        )
    return page


def _hook(page: Page | None, *, generate_parsed_pages: bool = False) -> None:
    options = ThreadedPdfPipelineOptions(generate_parsed_pages=generate_parsed_pages)
    stand_in = type("_P", (), {"pipeline_options": options})()
    PictureAwarePdfPipeline._release_page_resources(stand_in, _Item(payload=page))  # type: ignore[arg-type]


class TestTextOnlyPages:
    def test_everything_is_released(self) -> None:
        """A text page's images are never read again: assembly is the last consumer."""
        page = _page(DocItemLabel.TEXT, DocItemLabel.TABLE)
        backend = page._backend

        _hook(page)

        assert page._image_cache == {}
        assert page._backend is None
        assert backend.unloaded is True  # type: ignore[union-attr]
        assert page.predictions.layout is None
        assert page.parsed_page is None


class TestPicturePages:
    def test_only_the_crop_scale_survives(self) -> None:
        """Crops come from page.image, i.e. _default_image_scale. Every other render is dead
        weight even here — including the scale-2.0 one left by the table-structure stage on a
        page carrying both a table and a picture."""
        page = _page(DocItemLabel.TABLE, DocItemLabel.PICTURE)

        _hook(page)

        assert set(page._image_cache) == {1.5}

    def test_nothing_is_kept_when_the_crop_scale_was_never_rendered(self) -> None:
        page = _page(DocItemLabel.PICTURE)
        page._image_cache.pop(1.5)

        _hook(page)

        assert page._image_cache == {}

    def test_backend_is_unloaded_on_picture_pages_too(self) -> None:
        """What makes this competitive with Option E: with crops embedded on the document,
        nothing reads conv_res.pages after assembly, so no backend needs to survive."""
        page = _page(DocItemLabel.PICTURE)

        _hook(page)

        assert page._backend is None


class TestConservativePaths:
    def test_failed_page_keeps_its_images(self) -> None:
        """assembled is None means the page never completed; do not guess at what it holds."""
        page = _page(assembled=False)

        _hook(page)

        assert set(page._image_cache) == {1.0, 1.5, 2.0}
        assert page._backend is None  # still safe: nothing reads it after assembly

    def test_missing_payload_is_a_no_op(self) -> None:
        _hook(None)

    def test_parsed_page_is_kept_when_requested(self) -> None:
        page = _page(DocItemLabel.TEXT)
        page.parsed_page = object()  # type: ignore[assignment]

        _hook(page, generate_parsed_pages=True)

        assert page.parsed_page is not None


class TestFlagGuard:
    def test_lean_images_are_rejected_at_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unloading every backend is only safe while the crops are embedded on the document.
        With generate_picture_images off it would break classification silently, so it fails
        loudly here instead."""
        monkeypatch.setattr(StandardPdfPipeline, "__init__", lambda _self, _options: None)

        with pytest.raises(ValueError, match="generate_picture_images"):
            PictureAwarePdfPipeline(ThreadedPdfPipelineOptions(generate_picture_images=False))

        PictureAwarePdfPipeline(ThreadedPdfPipelineOptions(generate_picture_images=True))


class TestWiring:
    def test_parser_uses_the_releasing_pipeline(self) -> None:
        """The whole fix is inert unless the converter is built with it."""
        import inspect

        from src.services.ingestion import docling_parser

        source = inspect.getsource(docling_parser._create_converter)
        assert "pipeline_cls=PictureAwarePdfPipeline" in source

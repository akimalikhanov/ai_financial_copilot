"""Docling pipeline subclass that releases per-page memory the stock pipeline retains.

Docling frees a page's images and backend in `_release_page_resources`, but both releases are
gated on flags this application turns on: `keep_images` (set by `generate_picture_images`) and
`keep_backend` (set by `do_picture_classification`). With both on, nothing is ever released and
peak RSS grows with page count — measured at 13.1 MB/page, which OOM-kills a 12 GB worker at
~1000 pages. Docling has no narrower option: picture crops are minted at the very end of the
run, so it cannot know which pages will need their image until every page has been released.

The hook runs as the `assemble` stage's postprocess, after `page.assembled` is populated, so it
can make per page the decision Docling can only make globally: keep an image only on the pages
that actually hold a picture. Measured on a 1043-page filing: 15.7 GB peak stock, 7.3 GB here.

See docs/stages/ingestion-pipeline-findings.md §2 Option A for the measurements and the full
safety argument. All of this reads Docling private attributes and is pinned to docling 2.111.0.
"""

from __future__ import annotations

from typing import Any

from docling.datamodel.base_models import PagePredictions
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc.labels import DocItemLabel


class PictureAwarePdfPipeline(StandardPdfPipeline):
    """StandardPdfPipeline that keeps page images only where a picture will be cropped."""

    def __init__(self, pipeline_options: ThreadedPdfPipelineOptions) -> None:
        super().__init__(pipeline_options)
        if not pipeline_options.generate_picture_images:
            # Load-bearing, not a preference: with crops embedded on the document, enrichment
            # takes prepare_element's embedded-crop branch and never reads conv_res.pages —
            # which is what makes unloading every page backend below safe. Turn this off and
            # the unload silently breaks picture classification instead of failing here.
            raise ValueError(
                "PictureAwarePdfPipeline requires generate_picture_images=True "
                "(DOCLING_GENERATE_PICTURE_IMAGES); without it, unloading page backends "
                "breaks picture crops and classification"
            )

    def _release_page_resources(self, item: Any) -> None:
        page = getattr(item, "payload", None)
        if page is None:
            return

        # Unconditional: nothing reads either after assembly. The parent method would skip
        # both, since keep_images and keep_backend are true in this configuration — which is
        # why this does not delegate to super().
        page.predictions = PagePredictions()
        if page._backend is not None:
            page._backend.unload()
            page._backend = None
        if not self.pipeline_options.generate_parsed_pages:
            page.parsed_page = None

        if page.assembled is None:
            # A page that failed assembly: keep its images rather than guess.
            return
        if any(e.label == DocItemLabel.PICTURE for e in page.assembled.elements):
            # Crops are taken from `page.image`, i.e. _default_image_scale, so every other
            # render is dead weight even here: preprocessing's unconditional scale-1.0 pass,
            # and the scale-2.0 render the table-structure stage leaves on a page that holds
            # both a table and a picture.
            crop_scale = page._default_image_scale
            keep = page._image_cache.get(crop_scale)
            page._image_cache = {crop_scale: keep} if keep is not None else {}
        else:
            page._image_cache = {}

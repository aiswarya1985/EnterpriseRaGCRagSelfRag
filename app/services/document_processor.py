from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from loguru import logger
'''1. 
Initialization (__init__)
PdfPipelineOptions(...): Sets up PDF parsing to use Apple Silicon GPU (AcceleratorDevice.MPS) with 8 CPU threads for fast OCR/layout extraction.
DocumentConverter: Initializes the PDF converter engine.
HybridChunker: Loads a chunking strategy that splits documents intelligently using both layout (headers, paragraphs) and token length constraints.
2 chunks for simple doc
[
    {
        "text": "Docling is an open-source library for document processing.",
        "source": "sample.pdf",
        "page_number": 1
    },
    {
        "text": "It supports OCR, PDF layout analysis, and chunking.",
        "source": "sample.pdf",
        "page_number": 2
    }
]
'''
class DocumentProcessor:
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8, device=AcceleratorDevice.MPS
        )
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        self.chunker = HybridChunker()

    def process_document(self, file_path: str) -> list[dict]:
        result = self.converter.convert(file_path)
        doc = result.document
        chunk_iter = self.chunker.chunk(doc)

        chunks = []
        source_name = Path(file_path).name

        for chunk in chunk_iter:
            meta = {"text": chunk.text, "source": source_name}
            if hasattr(chunk, "meta") and hasattr(chunk.meta, "doc_items"):
                items = chunk.meta.doc_items
                if items and hasattr(items[0], "prov") and items[0].prov:
                    meta["page_number"] = items[0].prov[0].page_no
            chunks.append(meta)
        logger.info("Processed {} chunks from {}", len(chunks), file_path)
        return chunks

                
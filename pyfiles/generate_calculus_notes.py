# See README header in this script for usage.
# (Identical logic to the earlier version, with full‑document processing and exports.)
# To run: python generate_calculus_notes.py --input /path/to/pdfs --out out_dir
#!/usr/bin/env python3
"""
Calculus Notes Generator - PDF to Study Materials Converter

Author: Cazandra Aporbo
Purpose: Transform calculus PDF documents into structured study materials including
         organized notes, flashcards, and personalized study plans. Updated October 2025

Usage:
    python generate_calculus_notes.py --input /path/to/pdfs --out output_directory
    python generate_calculus_notes.py --input pdfs/ --out study_materials/ --fast

This tool processes calculus PDFs and generates:
- Structured markdown notes organized by topic
- CSV flashcards for spaced repetition
- JSON study plan with weekly goals
- Consolidated notes from multiple sources
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path


class PDFExtractor:
    """Handles PDF text extraction using multiple fallback methods"""

    @staticmethod
    def extract_with_pypdf2(pdf_path: Path, max_pages: int | None = None) -> str:
        """Extract text using PyPDF2 library"""
        try:
            import PyPDF2
        except ImportError:
            print(f"PyPDF2 not installed. Skipping {pdf_path.name}")
            return ""

        try:
            text_chunks = []
            with pdf_path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = reader.pages if max_pages is None else reader.pages[:max_pages]

                for page in pages:
                    text_chunks.append(page.extract_text() or "")

            return "\n".join(text_chunks)
        except Exception as e:
            print(f"PyPDF2 extraction failed for {pdf_path.name}: {e}")
            return ""

    @staticmethod
    def extract_with_pdfminer(pdf_path: Path) -> str:
        """Extract text using pdfminer library as fallback"""
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
        except ImportError:
            print("pdfminer not installed. Consider installing for better extraction.")
            return ""

        try:
            return pdfminer_extract(str(pdf_path)) or ""
        except Exception as e:
            print(f"pdfminer extraction failed for {pdf_path.name}: {e}")
            return ""

    @classmethod
    def extract_text(cls, pdf_path: Path, fast_mode: bool = False) -> str:
        """
        Extract text from PDF with fallback methods
        
        Args:
            pdf_path: Path to PDF file
            fast_mode: If True, only process first 12 pages for quick preview
        
        Returns:
            Extracted text as string
        """
        if fast_mode:
            # Try fast extraction with page limit
            text = cls.extract_with_pypdf2(pdf_path, max_pages=12)
            if text.strip():
                return text

            # Fallback to pdfminer but limit lines
            text = cls.extract_with_pdfminer(pdf_path)
            return "\n".join(text.splitlines()[:1200])

        # Full extraction
        text = cls.extract_with_pypdf2(pdf_path)
        if text.strip():
            return text

        return cls.extract_with_pdfminer(pdf_path)


class TextProcessor:
    """Processes and categorizes mathematical text content"""

    # Topic patterns for categorizing content
    TOPIC_PATTERNS = [
        (r"\b(limit|continuity|IVT|intermediate value|squeeze|epsilon-delta)\b",
         "Limits & Continuity"),

        (r"\b(derivative|differentiat|chain rule|product rule|quotient rule|tangent|critical point|implicit)\b",
         "Derivatives"),

        (r"\b(integral|antiderivative|FTC|fundamental theorem|substitution|parts|riemann|area under curve)\b",
         "Integrals"),

        (r"\b(series|sequence|convergence|divergence|ratio test|root test|alternating|taylor|maclaurin|power series)\b",
         "Series & Convergence"),

        (r"\b(maximiz|minimiz|optimization|gradient|concavity|inflection|relative extrema)\b",
         "Optimization & Analysis"),

        (r"\b(log|ln|exponential|exp|natural log|e\^)\b",
         "Exponential & Logarithmic"),

        (r"\b(differential equation|slope field|euler|separation of variables)\b",
         "Differential Equations"),

        (r"\b(vector|dot product|cross product|parametric|polar)\b",
         "Vectors & Parametric"),
    ]

    @staticmethod
    def normalize_text(text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace while preserving structure
        lines = []
        for line in text.splitlines():
            # Clean up spacing
            cleaned = re.sub(r"[ \t]+", " ", line.strip())

            # Skip lines that are just formatting characters
            if cleaned and not re.fullmatch(r"[-_=*\s]+", cleaned):
                lines.append(cleaned)

        return "\n".join(lines)

    @classmethod
    def split_into_sections(cls, text: str) -> dict[str, list[str]]:
        """
        Categorize text lines into mathematical topics
        
        Returns:
            Dictionary mapping topic names to lists of relevant lines
        """
        # Initialize sections
        sections = {name: [] for _, name in cls.TOPIC_PATTERNS}
        sections["General Notes"] = []

        # Process each line
        for line in text.split("\n"):
            if not line.strip():
                continue

            categorized = False
            for pattern, topic_name in cls.TOPIC_PATTERNS:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    sections[topic_name].append(line)
                    categorized = True
                    break

            if not categorized:
                sections["General Notes"].append(line)

        # Remove empty sections
        return {
            topic: lines
            for topic, lines in sections.items()
            if lines
        }

    @staticmethod
    def extract_flashcards(text: str) -> list[tuple[str, str]]:
        """
        Extract question-answer pairs from text for flashcards
        
        Returns:
            List of (question, answer) tuples
        """
        # Pattern for mathematical notation
        math_pattern = re.compile(
            r"(lim\b|∫|Σ|sum|d/dx|dx\b|dy\b|e\^[^\s]+|ln\([^)]*\)|exp\([^)]*\)|√|∞)",
            re.IGNORECASE
        )

        cards = []

        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) < 10:
                continue

            lower = line.lower()

            # Extract definitions and theorems
            if any(keyword in lower for keyword in ["rule", "theorem", "test", "definition"]):
                question = f"Define: {line.split(':')[0][:100]}"
                cards.append((question, line))

            # Extract mathematical expressions
            elif any(keyword in lower for keyword in ["limit", "derivative", "integral", "series"]):
                if math_pattern.search(line):
                    question = f"Evaluate or explain: {line[:120]}"
                    cards.append((question, line))

            # Extract procedures
            elif any(keyword in lower for keyword in ["find", "compute", "calculate", "solve"]):
                question = f"How to: {line[:100]}"
                cards.append((question, line))

        # Remove duplicates while preserving order
        seen = set()
        unique_cards = []
        for q, a in cards:
            if (q, a) not in seen:
                unique_cards.append((q, a))
                seen.add((q, a))

        return unique_cards[:400]  # Limit to manageable number


class StudyMaterialsGenerator:
    """Generates various study materials from processed text"""

    @staticmethod
    def render_markdown(title: str, sections: dict[str, list[str]], metadata: dict) -> str:
        """
        Generate formatted markdown notes
        
        Args:
            title: Document title
            sections: Categorized content
            metadata: Document metadata
        
        Returns:
            Formatted markdown string
        """
        markdown_lines = [
            f"# {title}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Source:** `{metadata.get('source', 'Unknown')}`",
            "",
            "## Table of Contents",
            ""
        ]

        # Add TOC
        for section_name in sections:
            markdown_lines.append(f"- [{section_name}](#{section_name.lower().replace(' ', '-').replace('&', '')})")

        markdown_lines.append("")

        # Add sections
        for section_name, lines in sections.items():
            markdown_lines.append(f"## {section_name}")
            markdown_lines.append("")

            # Limit lines per section for readability
            for line in lines[:500]:
                markdown_lines.append(f"- {line}")

            if len(lines) > 500:
                markdown_lines.append(f"\n*... and {len(lines) - 500} more items*")

            markdown_lines.append("")

        return "\n".join(markdown_lines)

    @staticmethod
    def build_study_plan() -> list[dict]:
        """
        Generate a structured 4-week study plan for calculus
        
        Returns:
            List of weekly study plans with goals and daily drills
        """
        return [
            {
                "week": 1,
                "focus": "Limits & Continuity",
                "goals": [
                    "Master limit evaluation techniques",
                    "Understand continuity definitions",
                    "Apply squeeze theorem",
                    "Solve epsilon-delta proofs"
                ],
                "daily_drills": [
                    "10 limit problems",
                    "5 continuity checks",
                    "2 epsilon-delta proofs"
                ],
                "resources": ["Khan Academy Limits", "Paul's Online Notes"]
            },
            {
                "week": 2,
                "focus": "Derivatives",
                "goals": [
                    "Master differentiation rules",
                    "Chain rule applications",
                    "Implicit differentiation",
                    "Find critical points"
                ],
                "daily_drills": [
                    "20 derivative problems",
                    "5 chain rule applications",
                    "3 optimization problems"
                ],
                "resources": ["3Blue1Brown Essence of Calculus", "MIT OpenCourseWare"]
            },
            {
                "week": 3,
                "focus": "Integrals",
                "goals": [
                    "Fundamental Theorem of Calculus",
                    "U-substitution mastery",
                    "Integration by parts",
                    "Trigonometric integrals"
                ],
                "daily_drills": [
                    "15 integration problems",
                    "5 u-substitutions",
                    "3 integration by parts"
                ],
                "resources": ["Professor Leonard YouTube", "Symbolab Practice"]
            },
            {
                "week": 4,
                "focus": "Series & Advanced Topics",
                "goals": [
                    "Convergence tests",
                    "Taylor series",
                    "Power series",
                    "Applications"
                ],
                "daily_drills": [
                    "10 series convergence tests",
                    "5 Taylor expansions",
                    "Review previous topics"
                ],
                "resources": ["PatrickJMT Series Playlist", "Wolfram Alpha"]
            }
        ]


def process_documents(input_dir: Path, output_dir: Path, fast_mode: bool = False):
    """
    Main processing function for converting PDFs to study materials
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for output files
        fast_mode: Whether to use fast extraction mode
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDF files
    pdf_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf"])

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    # Storage for consolidated data
    all_documents = []
    all_flashcards = []

    # Process each PDF
    extractor = PDFExtractor()
    processor = TextProcessor()
    generator = StudyMaterialsGenerator()

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        # Extract and process text
        raw_text = extractor.extract_text(pdf_path, fast_mode)
        if not raw_text:
            print(f"  Warning: No text extracted from {pdf_path.name}")
            continue

        normalized_text = processor.normalize_text(raw_text)
        sections = processor.split_into_sections(normalized_text)

        # Generate individual document notes
        title = f"Structured Notes - {pdf_path.stem}"
        metadata = {"source": pdf_path.name}
        markdown_content = generator.render_markdown(title, sections, metadata)

        output_file = output_dir / f"{pdf_path.stem}_notes.md"
        output_file.write_text(markdown_content, encoding="utf-8")
        print(f"  Created: {output_file.name}")

        # Extract flashcards
        flashcards = processor.extract_flashcards(normalized_text)
        for question, answer in flashcards:
            all_flashcards.append({
                "source": pdf_path.name,
                "question": question,
                "answer": answer
            })

        # Store for consolidation
        all_documents.append({
            "source": pdf_path.name,
            "sections": sections
        })

    # Generate consolidated materials
    print("\nGenerating consolidated study materials...")

    # Merge all sections
    merged_sections = {}
    for doc in all_documents:
        for section_name, lines in doc["sections"].items():
            if section_name not in merged_sections:
                merged_sections[section_name] = []
            merged_sections[section_name].extend(lines)

    # Remove duplicates from merged sections
    for section_name in merged_sections:
        merged_sections[section_name] = list(dict.fromkeys(merged_sections[section_name]))

    # Create consolidated notes
    consolidated_markdown = generator.render_markdown(
        "Consolidated Calculus Notes",
        merged_sections,
        {"source": "All documents"}
    )

    consolidated_file = output_dir / "consolidated_notes.md"
    consolidated_file.write_text(consolidated_markdown, encoding="utf-8")
    print(f"  Created: {consolidated_file.name}")

    # Save flashcards to CSV
    flashcards_file = output_dir / "flashcards.csv"
    with flashcards_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "question", "answer"])
        writer.writeheader()
        writer.writerows(all_flashcards)
    print(f"  Created: {flashcards_file.name} ({len(all_flashcards)} cards)")

    # Generate study plan
    study_plan = generator.build_study_plan()
    study_plan_file = output_dir / "study_plan.json"
    study_plan_file.write_text(json.dumps(study_plan, indent=2), encoding="utf-8")
    print(f"  Created: {study_plan_file.name}")

    print(f"\nProcessing complete! All materials saved to {output_dir}")


def main():
    """Command-line interface for the calculus notes generator"""
    parser = argparse.ArgumentParser(
        description="Convert calculus PDFs into structured study materials",
        epilog="Example: python generate_calculus_notes.py --input pdfs/ --out study/"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing PDF files to process"
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for generated study materials"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: process only first 12 pages of each PDF"
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input)
    output_dir = Path(args.out)

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return 1

    if not input_dir.is_dir():
        print(f"Error: '{input_dir}' is not a directory")
        return 1

    # Process documents
    try:
        process_documents(input_dir, output_dir, args.fast)
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

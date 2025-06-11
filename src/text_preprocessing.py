import re
import json
from bs4 import BeautifulSoup
from typing import Dict, Optional


class TenKPreprocessor:
    SECTION_PATTERN = re.compile(r"(ITEM\s+[\dA-Z\.]+)\s+([^\n]*)", re.IGNORECASE)

    def __init__(self):
        self.boilerplate_patterns = [
            "forward-looking statements",
            "check mark",
            "pursuant to section",
            "documents incorporated by reference",
            "the registrant",
            "see the definitions of"
        ]

    def preprocess(self, text: str, doc_id: Optional[str] = None) -> Dict:
        text = self.strip_html(text)
        text = self.clean_whitespace_and_symbols(text)
        text = self.remove_form_headers(text)
        text = self.remove_boilerplate_sections(text)
        sections = self.extract_sections(text)

        if not sections:
            return {
                "doc_id": doc_id,
                "sections": {
                    "FULL_DOCUMENT": text
                }
            }

        return {
            "doc_id": doc_id,
            "sections": sections
        }

    def strip_html(self, text: str) -> str:
        return BeautifulSoup(text, "lxml").get_text()

    def clean_whitespace_and_symbols(self, text: str) -> str:
        text = text.replace("\t", " ")
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = text.replace("•", "-")
        text = text.replace("þ", "")
        return text.strip()

    def remove_form_headers(self, text: str) -> str:
        lines = text.splitlines()
        return "\n".join(
            line for line in lines
            if not line.strip().startswith("#") and not line.strip().endswith("þ") and not "check mark" in line.lower()
        )

    def remove_boilerplate_sections(self, text: str) -> str:
        lines = text.splitlines()
        return "\n".join([
            line for line in lines
            if not any(p in line.lower() for p in self.boilerplate_patterns)
        ])

    def extract_sections(self, text: str) -> Dict[str, str]:
        matches = list(self.SECTION_PATTERN.finditer(text))
        sections = {}

        for i, match in enumerate(matches):
            section_title = match.group(1).strip().upper()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            sections[section_title] = content

        return sections


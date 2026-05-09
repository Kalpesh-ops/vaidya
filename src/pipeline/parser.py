# src/pipeline/parser.py

import re
import json
import unicodedata
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from loguru import logger
from tqdm import tqdm


@dataclass
class Verse:
    verse_id: str           # Unique ID: CS_SU_01_024
    sthana: str             # Sutrasthana
    sthana_num: int         # 1
    adhyaya: int            # Chapter number
    verse_num: int          # Verse number
    text_sanskrit: str      # Original Devanagari
    source: str             # priyavrata_sharma
    raw_line: str           # Original OCR line
    citation: str           # Human-readable: CS.Su.1.24


STHANA_MAP = {
    "सूत्रस्थानम्": ("Sutrasthana", 1),
    "सूत्रस्थान": ("Sutrasthana", 1),
    "निदानस्थानम्": ("Nidanasthana", 2),
    "निदानस्थान": ("Nidanasthana", 2),
    "विमानस्थानम्": ("Vimanasthana", 3),
    "विमानस्थान": ("Vimanasthana", 3),
    "शारीरस्थानम्": ("Sharirasthana", 4),
    "शारीरस्थान": ("Sharirasthana", 4),
    "इन्द्रियस्थानम्": ("Indriyasthana", 5),
    "इन्द्रियस्थान": ("Indriyasthana", 5),
    "चिकित्सास्थानम्": ("Cikitsasthana", 6),
    "चिकित्सास्थान": ("Cikitsasthana", 6),
    "कल्पस्थानम्": ("Kalpasthana", 7),
    "कल्पस्थान": ("Kalpasthana", 7),
    "सिद्धिस्थानम्": ("Siddhisthana", 8),
    "सिद्धिस्थान": ("Siddhisthana", 8),
}

STHANA_ABBREV = {
    "Sutrasthana": "Su",
    "Nidanasthana": "Ni",
    "Vimanasthana": "Vi",
    "Sharirasthana": "Sa",
    "Indriyasthana": "In",
    "Cikitsasthana": "Ci",
    "Kalpasthana": "Ka",
    "Siddhisthana": "Si",
}


class CarakaParser:
    def __init__(self, source_path: str, source_name: str = "priyavrata_sharma"):
        self.source_path = Path(source_path)
        self.source_name = source_name
        self.verses = []
        self.errors = []
        self.current_sthana = "Sutrasthana"
        self.current_sthana_num = 1
        self.current_adhyaya = 1

        # Verse number pattern: ।। २४।। or || 24 ||
        self.verse_pattern = re.compile(
            r'(.*?)[।|]\s*[।|]\s*(\d+|[०-९]+)\s*[।|]\s*[।|]'
        )
        # Adhyaya pattern
        self.adhyaya_pattern = re.compile(
            r'(?:प्रथम|द्वितीय|तृतीय|चतुर्थ|पञ्चम|षष्ठ|सप्तम|अष्टम|नवम|दशम|'
            r'एकादश|द्वादश|त्रयोदश|चतुर्दश|पञ्चदश|षोडश|सप्तदश|अष्टादश|'
            r'एकोनविंश|विंश|एकविंश|द्वाविंश|त्रयोविंश|चतुर्विंश|पञ्चविंश|'
            r'षड्विंश|सप्तविंश|अष्टाविंश|एकोनत्रिंश|त्रिंश)(?:ोऽध्याय|ोऽध्यायः|'
            r'अध्याय|अध्यायः)'
        )
        self.adhyaya_num_pattern = re.compile(r'अध्याय[ःं]?\s*$')

    def devanagari_to_int(self, s: str) -> int:
        devanagari_digits = str.maketrans('०१२३४५६७८९', '0123456789')
        return int(s.translate(devanagari_digits))

    def detect_sthana(self, line: str) -> bool:
        for key, (name, num) in STHANA_MAP.items():
            if key in line:
                self.current_sthana = name
                self.current_sthana_num = num
                self.current_adhyaya = 1
                logger.info(f"Detected sthana: {name}")
                return True
        return False

    def detect_adhyaya(self, line: str) -> bool:
        # Look for adhyaya markers
        ordinals = {
            'प्रथम': 1, 'द्वितीय': 2, 'तृतीय': 3, 'चतुर्थ': 4,
            'पञ्चम': 5, 'षष्ठ': 6, 'सप्तम': 7, 'अष्टम': 8,
            'नवम': 9, 'दशम': 10, 'एकादश': 11, 'द्वादश': 12,
            'त्रयोदश': 13, 'चतुर्दश': 14, 'पञ्चदश': 15, 'षोडश': 16,
            'सप्तदश': 17, 'अष्टादश': 18, 'एकोनविंश': 19, 'विंश': 20,
            'एकविंश': 21, 'द्वाविंश': 22, 'त्रयोविंश': 23,
            'चतुर्विंश': 24, 'पञ्चविंश': 25, 'षड्विंश': 26,
            'सप्तविंश': 27, 'अष्टाविंश': 28, 'एकोनत्रिंश': 29,
            'त्रिंश': 30,
        }
        for word, num in ordinals.items():
            if word in line and 'अध्याय' in line:
                self.current_adhyaya = num
                logger.info(f"Detected adhyaya: {num}")
                return True
        return False

    def extract_verse_num(self, raw_num: str) -> int:
        raw_num = raw_num.strip()
        try:
            if any('\u0966' <= c <= '\u096f' for c in raw_num):
                return self.devanagari_to_int(raw_num)
            return int(raw_num)
        except ValueError:
            return -1

    def make_verse_id(self, sthana_num, adhyaya, verse_num) -> str:
        return f"CS_{sthana_num:02d}_{adhyaya:03d}_{verse_num:04d}"

    def make_citation(self, sthana, adhyaya, verse_num) -> str:
        abbrev = STHANA_ABBREV.get(sthana, "??")
        return f"CS.{abbrev}.{adhyaya}.{verse_num}"

    def clean_text(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        # Remove OCR artifacts
        text = re.sub(r'[^\u0900-\u097F\u0020-\u007E।॥\s]', '', text)
        return text.strip()

    def parse(self) -> list:
        logger.info(f"Parsing: {self.source_path}")

        with open(self.source_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_verse_lines = []
        
        for i, line in enumerate(tqdm(lines, desc="Parsing")):
            line = line.strip()
            if not line:
                continue

            # Check for sthana change
            self.detect_sthana(line)
            # Check for adhyaya change
            self.detect_adhyaya(line)

            # Try to match verse pattern
            match = self.verse_pattern.search(line)
            if match:
                verse_text = match.group(1).strip()
                verse_num_raw = match.group(2)
                verse_num = self.extract_verse_num(verse_num_raw)

                if verse_num == -1:
                    self.errors.append({
                        "line": i,
                        "raw": line,
                        "error": "Could not parse verse number"
                    })
                    continue

                # Combine with accumulated lines if any
                if current_verse_lines:
                    verse_text = ' '.join(current_verse_lines) + ' ' + verse_text
                    current_verse_lines = []

                verse_text = self.clean_text(verse_text)

                if len(verse_text) < 5:
                    continue

                verse = Verse(
                    verse_id=self.make_verse_id(
                        self.current_sthana_num,
                        self.current_adhyaya,
                        verse_num
                    ),
                    sthana=self.current_sthana,
                    sthana_num=self.current_sthana_num,
                    adhyaya=self.current_adhyaya,
                    verse_num=verse_num,
                    text_sanskrit=verse_text,
                    source=self.source_name,
                    raw_line=line,
                    citation=self.make_citation(
                        self.current_sthana,
                        self.current_adhyaya,
                        verse_num
                    )
                )
                self.verses.append(verse)
            else:
                # Accumulate multi-line verses
                if line and not any(k in line for k in STHANA_MAP):
                    current_verse_lines.append(line)
                    if len(current_verse_lines) > 5:
                        current_verse_lines = []

        logger.info(f"Parsed {len(self.verses)} verses, {len(self.errors)} errors")
        return self.verses

    def save(self, output_path: str):
        output = {
            "source": self.source_name,
            "total_verses": len(self.verses),
            "total_errors": len(self.errors),
            "verses": [asdict(v) for v in self.verses],
            "errors": self.errors
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to {output_path}")


# Stage 1 Benchmark
class ParserBenchmark:
    def __init__(self, parsed_verses: list):
        self.verses = parsed_verses

    def run(self) -> dict:
        results = {}

        # 1. Total verses extracted
        results["total_verses"] = len(self.verses)

        # 2. Citation completeness
        complete_citations = sum(
            1 for v in self.verses
            if v["citation"] and "??" not in v["citation"]
        )
        results["citation_completeness"] = complete_citations / len(self.verses)

        # 3. Sthana distribution
        sthana_dist = {}
        for v in self.verses:
            s = v["sthana"]
            sthana_dist[s] = sthana_dist.get(s, 0) + 1
        results["sthana_distribution"] = sthana_dist

        # 4. Average verse length
        avg_len = sum(len(v["text_sanskrit"]) for v in self.verses) / len(self.verses)
        results["avg_verse_length_chars"] = round(avg_len, 2)

        # 5. Empty verse check
        empty = sum(1 for v in self.verses if len(v["text_sanskrit"]) < 10)
        results["empty_or_short_verses"] = empty

        # 6. Devanagari integrity
        def is_devanagari(text):
            deva_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
            return deva_chars / max(len(text), 1)

        avg_deva = sum(
            is_devanagari(v["text_sanskrit"]) for v in self.verses
        ) / len(self.verses)
        results["devanagari_ratio"] = round(avg_deva, 4)

        # 7. Pass/Fail
        results["PASS"] = (
            results["citation_completeness"] > 0.95 and
            results["devanagari_ratio"] > 0.70 and
            results["empty_or_short_verses"] < len(self.verses) * 0.05
        )

        return results


if __name__ == "__main__":
    import sys

    source_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/priyavrata_sharma.txt"

    parser = CarakaParser(source_file, "priyavrata_sharma")
    verses = parser.parse()
    parser.save("data/processed/parsed_verses.json")

    # Run benchmark
    with open("data/processed/parsed_verses.json") as f:
        data = json.load(f)

    bench = ParserBenchmark(data["verses"])
    results = bench.run()

    print("\n=== STAGE 1 BENCHMARK RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    status = "✓ PASS" if results["PASS"] else "✗ FAIL"
    print(f"\nOverall: {status}")

    # Save benchmark report
    with open("benchmark_report.md", "w") as f:
        f.write("# Vaidya Benchmark Report\n\n")
        f.write("## Stage 1: Parser\n\n")
        for k, v in results.items():
            f.write(f"- **{k}**: {v}\n")
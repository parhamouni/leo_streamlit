#!/usr/bin/env python3
"""
Deep debug of find_phrase_matches function
"""
import os
import sys
from pathlib import Path
import fitz

sys.path.insert(0, str(Path(__file__).parent))

import utils_ade as ade

# Test PDF
pdf_path = Path("subset_gold/selected_pages_no_annotations.pdf")

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

doc = fitz.open(stream=pdf_bytes, filetype="pdf")
page = doc[0]

# Get tokens
tokens = ade.get_pdf_text_tokens(page)

print(f"Total tokens: {len(tokens)}")
print(f"\nSearching for 'keynote'...")

# Find tokens containing "keynote"
keynote_tokens = [t for t in tokens if "keynote" in t["text"].lower()]
print(f"Found {len(keynote_tokens)} tokens with 'keynote':")
for t in keynote_tokens:
    print(f"  '{t['text']}' @ ({t['x0']:.1f}, {t['y0']:.1f})")

# Now test find_phrase_matches
print(f"\nTesting find_phrase_matches('keynote')...")
matches = ade.find_phrase_matches(tokens, "keynote", None)
print(f"Result: {len(matches)} matches")

# Test with exact text
print(f"\nTesting find_phrase_matches('KEYNOTES')...")
matches2 = ade.find_phrase_matches(tokens, "KEYNOTES", None)
print(f"Result: {len(matches2)} matches")

# Test canonical
print(f"\nCanonical forms:")
print(f"  _canon('keynote') = '{ade._canon('keynote')}'")
print(f"  _canon('KEYNOTES') = '{ade._canon('KEYNOTES')}'")
print(f"  _canon('KEY NOTES') = '{ade._canon('KEY NOTES')}'")

doc.close()




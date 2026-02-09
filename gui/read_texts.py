import json
from pathlib import Path
import re
text = Path('binary_dataset_builder.py').read_text(encoding='utf-8')
match = re.search(r"return json.loads\('''\n(.*?)\n\s*'''\)", text, re.S)
if not match:
    raise RuntimeError('JSON block not found')
json_block = match.group(1)
print(repr(json_block[:200]))

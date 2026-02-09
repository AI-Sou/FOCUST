from pathlib import Path
text = """
PLACEHOLDER
"""
Path('batch_eval_replacement.txt').write_text(text.lstrip("\n"), encoding='utf-8')

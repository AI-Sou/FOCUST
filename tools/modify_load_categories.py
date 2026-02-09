from pathlib import Path

path = Path('binary_dataset_builder.py')
text = path.read_text(encoding='utf-8')
old_segment = "        self.categories = self.builder.load_categories()\n        if not self.categories:\n            return\n"
new_segment = "        self.categories = self.builder.load_categories()\n        if not self.categories:\n            self._refresh_category_labels()\n            return\n"
if old_segment not in text:
    raise RuntimeError('load_categories early block not found')
text = text.replace(old_segment, new_segment, 1)
marker = "            self.category_table.setItem(i, 3, QTableWidgetItem(str(annotations)))\r\n    \r\n"
if marker not in text:
    raise RuntimeError('marker for refresh not found')
text = text.replace(marker, marker + "        self._refresh_category_labels()\n\n", 1)
path.write_text(text, encoding='utf-8')

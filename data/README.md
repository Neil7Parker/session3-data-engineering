# Instructor dataset

This project includes the instructor-provided `imdb_top_500.csv`, retrieved on 27 November 2026 from the course repository's `session-400/datasets/` directory.

- Source: https://gitee.com/lundechen/data_engineering_master_2026_fall/blob/master/session-400/datasets/imdb_top_500.csv
- Source Git blob SHA: `7a905418847290a5a593691b714f869318f95bb3`
- File SHA-256: `23b5d6f040bcf36b648ed8fedb4cad57a6713f2b53abf07a29e947c34486178e`
- Original size: 639,390 bytes; 500 rows, with 250 positive and 250 negative labels.
- Columns: `text`, `label`, `rating`; only `text` is a model feature. The original rating is not used, to avoid target leakage.
- The source Git blob checksum was independently verified after retrieval.

The loader removes one duplicate normalized review before splitting, leaving 499 unique reviews (249 negative and 250 positive). The original CSV is unchanged.

The default schema recognises `review`, `text` or `content` for review text, and `sentiment`, `label` or `target` for the binary label.
Accepted labels are `positive` / `negative`, `pos` / `neg`, or `1` / `0` (positive is 1).
Files must use UTF-8 or UTF-8 with BOM. Each label needs at least five unique reviews.
Training fails on missing data, ambiguous schema, invalid rows, or conflicting duplicate labels.
If the instructor uses different column names, set the `TEXT_COLUMN` and `LABEL_COLUMN` repository variables.

Keep the instructor's original file unchanged. The loader removes duplicate cleaned reviews in memory before splitting.

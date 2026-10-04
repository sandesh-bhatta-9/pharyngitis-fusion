# Data

The raw data are not stored in this repository. Download them with:

```bash
bash scripts/download_data.sh
```

This fetches `PGUPharyngitis.zip` from Figshare
([10.6084/m9.figshare.28163513](https://doi.org/10.6084/m9.figshare.28163513.v1)), checks its MD5 checksum and unpacks it into
`data/raw/`:

```
data/raw/excel.xlsx                         one row per patient: ID, age, sex, 20 symptoms, Diagnosis #1-#9
data/raw/data_image_pharyngitis_nature/     one folder per patient: <ID>/<ID>.jpg
```

The dataset is released under CC BY 4.0. If you use it, cite the data paper:

> Shojaei N, Rostami H, Barzegar M, et al. A publicly available pharyngitis dataset and baseline evaluations for
> bacterial or nonbacterial classification. *Scientific Data* (2025). https://doi.org/10.1038/s41597-025-05780-5

## What the code does with it

`python -m src.prepare_data` writes `data/processed/` (resized images, `clean.csv`, `columns.json` and an audit
report, `audit.md`). Notes from the audit, all handled in code and configuration:

- 742 patients in the table; patient 666 has no image.
- Patients 19/23, 218/256 and 301/305 share an identical photo but have different records, so all six are excluded
  (`columns.exclude_ids` in `configs/default.yaml`). 735 patients remain.
- Some image files are zero-padded (`001.jpg`) and three are named by timestamp; images are matched by their folder name.
- There is no city or phone column. The phone model is read from each photo's EXIF data (Samsung SM-G998B, Xiaomi
  2201117SG, or unknown).
- 100 patients have a tied physician vote. The main analysis counts ties as bacterial; `configs/tie_nonbacterial.yaml`
  and `configs/tie_drop.yaml` run the alternatives.

Only the fold assignments (`data/processed*/folds.csv`) are committed, so every model in this repository and in future
work can use the same splits.

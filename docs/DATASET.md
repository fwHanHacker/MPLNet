# Dataset Preparation

MPLNet expects TuSimple-style JSON-lines annotations. Each line is a JSON object containing at least:

- `raw_file`: relative image path, usually under `clips/`.
- `lanes`: a list of x-coordinate lists.
- `h_samples`: y-coordinate samples shared by all lanes.

The default config paths assume:

```text
mfplnet/data/<dataset_name>/
|-- clips/
|-- seg_label/
|-- label_data_0114.json
|-- label_data_0115.json
`-- ...
```

Segmentation masks are resolved by replacing `clips` in `raw_file` with `seg_label` and changing the extension to `.png`.

## Built-In Dataset Splits

| Dataset class | Default root | Train files | Val files | Test files |
| --- | --- | --- | --- | --- |
| `Cpld01` | `mfplnet/data/cpld_01` | `label_data_0114.json` | `label_data_0115.json` | `label_data_0116.json` |
| `Pldu01` | `mfplnet/data/pldu_01` | `label_data_0114.json`, `label_data_0115.json` | `label_data_0116.json`, `label_data_0117.json` | `label_data_0118.json` |
| `Pldm01` | `mfplnet/data/pldm_01` | `label_data_0114.json` to `label_data_0117.json` | `label_data_0118.json` to `label_data_0121.json` | `label_data_0122.json` |

## Tools

- `tools/lable_to_dataset.py`: converts label files to the expected dataset layout.
- `tools/generate_seg.py`: generates or prepares segmentation labels.

Review the paths in each script before running them on a new machine.

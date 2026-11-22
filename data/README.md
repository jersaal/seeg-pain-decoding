# Data

The de-identified processed data are available from the corresponding author
upon reasonable request, as stated in the manuscript. They contain only what
the analysis uses: no dates, no clock times and no identifiers other than
the participant labels of the manuscript.

Copy the participant folders here (or point `data_dir` in `config.yaml`
elsewhere):

```
data/
├── P1/
│   ├── surveys.csv
│   ├── power_during_survey.csv
│   ├── power_pre_survey.csv
│   └── contacts.csv
├── P2/
│   └── ...
└── P6/
```

The three survey-level tables of a participant have one row per pain survey,
in chronological order, and their rows correspond.

## `surveys.csv`

| Column | Content |
|---|---|
| `day` | Recording day of the survey, counted from the first day (1, 2, ...); defines the cross-validation groups |
| `seconds` | Time of the survey in seconds from the participant's first survey |
| `pain_vas` | Pain intensity rating, visual analog scale 0–100 |

## `power_during_survey.csv` and `power_pre_survey.csv`

Spectral power in the 30 s after survey onset (`during`) and in the 30 s
window starting 60 s before it (`pre`). One column per gray-matter contact
and frequency band, named `<contact>_<band>`, e.g. `LACC4_high_gamma`: the
mean multitaper spectral power of the contact in `delta` (0.5–4 Hz), `theta`
(4–8 Hz), `alpha` (8–12 Hz), `beta` (12–30 Hz), `low_gamma` (30–70 Hz) or
`high_gamma` (70–150 Hz).

Surveys within 5 minutes of stimulation and epochs failing artifact rejection
were removed before these files were written (see Methods).

## `contacts.csv`

One row per gray-matter contact, in the order of the power columns.

| Column | Content |
|---|---|
| `contact` | Contact name, the prefix of its power columns |
| `region` | Anatomical region with hemisphere prefix, e.g. `L_MCC` |
| `MNI_x`, `MNI_y`, `MNI_z` | MNI152 coordinate in mm |

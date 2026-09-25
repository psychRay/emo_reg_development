# Figure 1

Run from the project root:

```bash
python run.py --figure 1 --toy
python run.py --figure 1 --input path/to/fig01_behavior.csv --output outputs/fig01/real
```

Real input is a CSV with one row per participant and condition. Required
columns:

| Column | Contents |
|---|---|
| `sub_id` | Participant ID; exactly three rows per participant |
| `age` | Age in years |
| `gender` | Gender group (`0`/`1`) |
| `site_id` | Recruitment site |
| `site_region` | `EC`, `MC`, `NC`, `SC`, `WC`, or full region name |
| `cognition` | `rpsl`, `lkng`, or `lknt` |
| `emot_rating` | Participant rating for that condition |

Supply analysis-ready participants with all three conditions. Outputs are saved
under `outputs/fig01/{toy,real}/`.

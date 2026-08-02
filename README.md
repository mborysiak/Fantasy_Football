# Fantasy_Football

## Production refresh

After manually loading the provider exports through
`Scripts/Data_Generation/1_Update_Projections.py`, run the complete downstream
build in an isolated stage:

```powershell
$env:FF_CURRENT_SEASON = '2026'
.\.venv_ff_312\Scripts\python.exe -m Scripts.V2.refresh_production --year 2026
```

The command does not change live model or app databases unless it is resumed
with the explicit `--promote` flag. The requested year must have a reviewed
entry in `Scripts/V2/production_cycle.py`; 2026 is the only approved current
cycle today. See
[`docs/runbooks/production_refresh.md`](docs/runbooks/production_refresh.md)
for the full operator workflow and release gates.

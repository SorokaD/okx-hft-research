import json
import pathlib

p = pathlib.Path(r"D:/tumar/okx-hft-research/notebooks/research/005_eda_targets_predictability.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
cell = nb["cells"][2]
s = "".join(cell["source"])
anchor = "\n\n    # keep the same preparation logic: median imputation\n"
guard = "\n    # Guard against invalid CatBoost params combinations in stale kernels/notebook states.\n    if cat_params.get(\"bootstrap_type\") != \"Bayesian\":\n        cat_params.pop(\"bagging_temperature\", None)\n"
if guard not in s:
    s = s.replace(anchor, guard + anchor)
    cell["source"] = s.splitlines(keepends=True)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("guard_added")
else:
    print("guard_exists")

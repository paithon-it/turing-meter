"""Check for invoice-total-001: exit 0 if the result matches the expected values."""
import json
import sys

expected = json.load(open("expected.json"))
try:
    got = json.load(open(sys.argv[1]))
except (OSError, ValueError) as err:
    print(f"cannot read result: {err}")
    sys.exit(1)

ok = (str(got.get("numero", "")).strip() == expected["numero"]
      and str(got.get("data", "")).strip() == expected["data"]
      and abs(float(got.get("totale_eur", -1)) - expected["totale_eur"]) < 0.005)
print("ok" if ok else f"mismatch: {got}")
sys.exit(0 if ok else 1)

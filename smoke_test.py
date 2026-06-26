import subprocess, sys, pathlib, shutil
root = pathlib.Path(__file__).resolve().parents[1]
out = root / "_smoke_output"
if out.exists():
    shutil.rmtree(out)
cmd = [sys.executable, str(root / "ccqm.py"), "run", str(root / "examples" / "quick_start.txt"), "--output", str(out), "--precision", "quick", "--no-plots"]
subprocess.check_call(cmd)
assert (out / "results.json").exists()
assert (out / "report.txt").exists()
assert (out / "tables" / "form_factors.csv").exists()
print("smoke test passed")

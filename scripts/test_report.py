"""Turn pytest JUnit XML files into the pass/fail tables for docs/test_report.md.

    pytest -m "not e2e"             --junitxml=reports/junit_offline.xml
    pytest -m "e2e and not drift"   --junitxml=reports/junit_e2e.xml
    pytest -m drift                 --junitxml=reports/junit_drift.xml
    python scripts/test_report.py reports/junit_*.xml > reports/test_results.md

Each test is mapped to its case ID in docs/test_cases.md from its name
(test_uf03_... -> UT-F-03). Parametrised tests count once per parameter set,
and the case table shows how many of each case's variants passed.
"""
import re
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict

PREFIX = OrderedDict([
    ("uf", ("UT-F", "Unit: feature engineering (features.py)")),
    ("up", ("UT-P", "Unit: PGN parsing (prepare.py, Stockfish_train_eval.py)")),
    ("um", ("UT-M", "Unit: model inputs and network (model.py)")),
    ("us", ("UT-S", "Unit: serving wrapper (pyfunc_model.py)")),
    ("ip", ("IT-P", "Integration: Stockfish -> prepare")),
    ("is", ("IT-S", "Integration: MLflow packaging")),
    ("dq", ("DQ", "Data quality (pipeline outputs)")),
    ("ac", ("AC", "Acceptance criteria (model)")),
    ("api", ("API", "API (live backend)")),
    ("e2e", ("E2E", "End to end (full stack)")),
])
NAME_RE = re.compile(r"test_(" + "|".join(PREFIX) + r")(\d+)_")


def outcome(case):
    for tag, result in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
        node = case.find(tag)
        if node is not None:
            return result, (node.get("message") or "").strip().splitlines()[:1]
    return "passed", []


def main(paths):
    by_case = OrderedDict()
    for path in paths:
        for tc in ET.parse(path).getroot().iter("testcase"):
            m = NAME_RE.search(tc.get("name", ""))
            if not tc.get("classname") and tc.find("error") is not None:
                # pytest records a module that failed to import as a nameless error
                cid, group = f"COLLECTION ERROR: {tc.get('name')}", "other"
                print(f"WARNING: {tc.get('name')} could not be collected; that run was "
                      "interrupted, so its other tests did not execute.\n", file=sys.stderr)
            elif m:
                code, _ = PREFIX[m.group(1)]
                cid = f"{code}-{int(m.group(2)):02d}"
                group = m.group(1)
            else:
                cid, group = tc.get("name"), "other"
            res, msg = outcome(tc)
            by_case.setdefault(cid, {"group": group, "results": [], "msgs": []})
            by_case[cid]["results"].append(res)
            by_case[cid]["msgs"] += msg

    totals = defaultdict(lambda: defaultdict(int))
    for cid, c in by_case.items():
        for r in c["results"]:
            totals[c["group"]][r] += 1
            totals["ALL"][r] += 1

    cols = ("passed", "failed", "error", "skipped")
    print("## Summary (test executions)\n")
    print("| Suite | Total | " + " | ".join(c.capitalize() for c in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for key, (code, label) in list(PREFIX.items()) + [("other", ("", "Other")), ("ALL", ("", "**Total**"))]:
        t = totals.get(key)
        if not t:
            continue
        print(f"| {label} | {sum(t.values())} | " + " | ".join(str(t[c]) for c in cols) + " |")

    executed = totals["ALL"]["passed"] + totals["ALL"]["failed"] + totals["ALL"]["error"]
    if executed:
        print(f"\nPass rate (excluding skipped): {totals['ALL']['passed'] / executed:.1%}")

    print("\n## Per test case\n")
    print("| Case | Result | Variants passed | Note |")
    print("|---|---|---|---|")
    for cid, c in by_case.items():
        rs = c["results"]
        run = [r for r in rs if r != "skipped"]
        verdict = ("SKIPPED" if not run else
                   "PASS" if all(r == "passed" for r in run) else "FAIL")
        note = c["msgs"][0].replace("|", "/")[:100] if c["msgs"] and verdict != "PASS" else ""
        print(f"| {cid} | {verdict} | {sum(r == 'passed' for r in rs)}/{len(rs)} | {note} |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])

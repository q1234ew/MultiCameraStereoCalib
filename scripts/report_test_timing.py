"""Summarize reported JUnit cases without changing pytest's release gate."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def seconds(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('JUnit durations must be finite and nonnegative')
    return result


def summarize(path, limit=20):
    if limit < 1:
        raise ValueError('Timing report limit must be positive')
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else root.findall('testsuite')
    if root.tag not in {'testsuite', 'testsuites'}:
        raise ValueError('Expected a JUnit testsuite or testsuites document')
    cases = list(root.iter('testcase'))
    counts = Counter()
    modules = defaultdict(lambda: dict(cases=0, seconds=0.))
    slowest, identities = [], []
    for case in cases:
        status = next((name for name in ('error', 'failure', 'skipped')
                       if case.find(name) is not None), 'passed')
        counts[status] += 1
        module, name = case.get('classname', ''), case.get('name', '')
        duration = seconds(case.get('time', '0'))
        modules[module]['cases'] += 1
        modules[module]['seconds'] += duration
        identities.append((module, name))
        slowest.append(dict(test=f'{module}::{name}', seconds=duration))
    return dict(
        status='failed' if counts['error'] or counts['failure'] else
               'passed' if cases else 'no_tests',
        reported_cases=len(cases), passed=counts['passed'], failures=counts['failure'],
        errors=counts['error'], skipped=counts['skipped'],
        collection_sha256=hashlib.sha256(json.dumps(sorted(identities),
            ensure_ascii=True).encode('ascii')).hexdigest(),
        junit_elapsed_seconds=round(sum(seconds(s.get('time', '0')) for s in suites), 3),
        reported_test_seconds=round(sum(row['seconds'] for row in slowest), 3),
        slowest_tests=sorted(slowest, key=lambda row: (-row['seconds'], row['test']))[:limit],
        slowest_modules=[dict(module=name, cases=row['cases'], seconds=round(row['seconds'], 3))
            for name, row in sorted(modules.items(), key=lambda item: (-item[1]['seconds'], item[0]))
            ][:limit],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('junit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path)
    args = parser.parse_args(argv)
    report = summarize(args.junit)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding='utf-8')
    if args.summary:
        with args.summary.open('a', encoding='utf-8') as stream:
            stream.write('\n### Regression timing\n\n')
            stream.write('Reported case times include setup/cleanup; parallel case totals are '
                         'not wall time. Pytest exit status remains the release gate.\n\n')
            stream.write('```json\n' + payload + '```\n')
    print(f"Reported tests: {report['reported_cases']}; JUnit elapsed: "
          f"{report['junit_elapsed_seconds']}s; report: {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

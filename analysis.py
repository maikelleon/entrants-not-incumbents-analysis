"""Exposure versus usage as predictors of young workers' employment, 2022-2025.

Compares occupation-level indices of theoretical AI exposure with usage-based
indices as predictors of the change in the share of workers aged 20 to 24 in
US occupations between the 2022 and 2025 annual averages of the Current
Population Survey (BLS, table 11b).

Inputs (all public; see README.md for sources):
  raw/cps11b_leaf_2022_2025.txt   title|total22|age20_24_22|age25plus22|total25|age20_24_25|age25plus25 (thousands)
  raw/cps11b_leaf_YYYY.txt        total,age20_24,age25plus for 2020, 2021, 2023, 2024, same row order
  raw/census2018_occ_soc.txt      Census 2018 occupation code|SOC code or pattern|title
  raw/job_exposure_compact.txt    SOC code,observed exposure (Anthropic Economic Index)
  data/occ_level.csv              Eloundou et al. exposure by O*NET-SOC code
  data/national_May2021_dl.csv    OEWS May 2021 national employment by SOC code
  data/ai_applicability_scores.csv  Tomlinson et al. AI applicability by SOC code
"""
import csv
import json
import math
import re
import sys
from collections import defaultdict

import numpy as np

RAW = 'raw/'
DATA = 'data/'


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


# ---------------------------------------------------------------- CPS outcomes
cps = []
for line in open(RAW + 'cps11b_leaf_2022_2025.txt', encoding='utf-8'):
    p = line.rstrip('\n').split('|')
    cps.append(dict(title=p[0], t22=int(p[1]), y22=int(p[2]), o22=int(p[3]),
                    t25=int(p[4]), y25=int(p[5]), o25=int(p[6])))

# ------------------------------------------------- Census occupation list
census = []
for line in open(RAW + 'census2018_occ_soc.txt', encoding='utf-8'):
    code, soc, title = line.rstrip('\n').split('|', 2)
    census.append(dict(code=code, soc=soc, title=title))
by_title = {norm(c['title']): c for c in census}
unmatched = [r['title'] for r in cps if norm(r['title']) not in by_title]
if unmatched:
    sys.exit('Unmatched CPS titles: %s' % unmatched)
for r in cps:
    c = by_title[norm(r['title'])]
    r['census'] = c['code']
    r['soc_pattern'] = c['soc']


def prefix(pattern):
    """SOC prefix covered by a Census code: exact detailed code, broad or
    minor group (trailing zeros), or residual group (X wildcards)."""
    s = pattern.replace('-', '')
    if 'X' in s:
        return s[:s.index('X')]
    return s.rstrip('0')


census_prefix = {c['code']: prefix(c['soc']) for c in census if c['soc'] != 'none'}


def assign(soc6):
    """Census code whose SOC prefix is the longest match for a detailed SOC code."""
    s = soc6.replace('-', '')
    best, best_len = None, -1
    for code, pre in census_prefix.items():
        if s.startswith(pre) and len(pre) > best_len:
            best, best_len = code, len(pre)
    return best


# --------------------------------------------------- employment weights (OEWS)
emp = {}
for row in csv.DictReader(open(DATA + 'national_May2021_dl.csv', encoding='utf-8-sig')):
    if row['O_GROUP'] == 'detailed':
        try:
            emp[row['OCC_CODE']] = float(row['TOT_EMP'].replace(',', ''))
        except ValueError:
            pass

# ----------------------------------------------------------- exposure indices
measures = {}

# Theoretical exposure (Eloundou et al.): O*NET-SOC codes averaged to SOC.
acc = defaultdict(lambda: defaultdict(list))
for row in csv.DictReader(open(DATA + 'occ_level.csv', encoding='utf-8')):
    soc = row['O*NET-SOC Code'][:7]
    acc['theory_human'][soc].append(float(row['human_rating_beta']))
    acc['theory_gpt4'][soc].append(float(row['dv_rating_beta']))
for k in acc:
    measures[k] = {s: sum(v) / len(v) for s, v in acc[k].items()}

# Observed exposure (Anthropic Economic Index, labor market impacts release).
measures['usage_claude'] = {}
for tok in open(RAW + 'job_exposure_compact.txt', encoding='utf-8').read().split():
    soc, val = tok.split(',')
    measures['usage_claude'][soc] = float(val)

# AI applicability (Tomlinson et al., Bing Copilot conversations).
measures['usage_copilot'] = {}
for row in csv.DictReader(open(DATA + 'ai_applicability_scores.csv', encoding='utf-8')):
    measures['usage_copilot'][row['SOC Code']] = float(row['ai_applicability_score'])

# Aggregate each index to Census occupations, weighting SOC codes by employment.
cen_measure = {}
coverage = {}
for name, m in measures.items():
    num, den, n_soc, n_noemp = defaultdict(float), defaultdict(float), 0, 0
    for soc, x in m.items():
        code = assign(soc)
        if code is None:
            continue
        w = emp.get(soc)
        if w is None:
            n_noemp += 1
            w = 1.0  # rare: detailed code without OEWS estimate gets minimal weight
        num[code] += w * x
        den[code] += w
        n_soc += 1
    cen_measure[name] = {c: num[c] / den[c] for c in num}
    coverage[name] = dict(soc_codes=n_soc, soc_without_oews=n_noemp,
                          census_codes=len(num))

# ------------------------------------------------------- panel of CPS years
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
for yr in (2020, 2021, 2023, 2024):
    vals = open(RAW + 'cps11b_leaf_%d.txt' % yr, encoding='utf-8').read().split()
    assert len(vals) == len(cps)
    for r, v in zip(cps, vals):
        t, y_, o = map(int, v.split(','))
        r['t%d' % (yr % 100)], r['y%d' % (yr % 100)], r['o%d' % (yr % 100)] = t, y_, o

names = ['theory_human', 'theory_gpt4', 'usage_claude', 'usage_copilot']

rows, dropped = [], dict(no_measure=0, small=0, reclassified=0)
for r in cps:
    if not all(r['census'] in cen_measure[n] for n in names):
        dropped['no_measure'] += 1
        continue
    if min(r['t%d' % (yr % 100)] for yr in YEARS) < 25:
        dropped['small'] += 1
        continue
    if max(abs(math.log(r['t%d' % (yr % 100)] / r['t22'])) for yr in YEARS) > math.log(2):
        dropped['reclassified'] += 1  # employment halved or doubled: likely coding change
        continue
    rr = dict(r)
    for n in names:
        rr[n] = cen_measure[n][r['census']]
    for yr in YEARS:
        k = yr % 100
        rr['s%d' % k] = 100 * r['y%d' % k] / r['t%d' % k]  # young share, percent
    rr['major'] = r['soc_pattern'][:2]
    rows.append(rr)

N = len(rows)
w = np.array([r['t22'] for r in rows], float)


def wstd(x, w):
    m = np.sum(w * x) / np.sum(w)
    sd = math.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w))
    return (x - m) / sd


Z = {n: wstd(np.array([r[n] for r in rows]), w) for n in names}
S = {yr: np.array([r['s%d' % (yr % 100)] for r in rows]) for yr in YEARS}
major = [r['major'] for r in rows]
no15 = np.array([m != '15' for m in major])


def wls(y, Xcols, w, fe=None):
    """Weighted least squares with HC1 standard errors. Returns coefficients
    and standard errors for Xcols (constant and fixed effects omitted) and
    the weighted R-squared."""
    X = [np.ones_like(y)] + list(Xcols)
    if fe is not None:
        for lv in sorted(set(fe))[1:]:
            X.append(np.array([1.0 if f == lv else 0.0 for f in fe]))
    X = np.column_stack(X)
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    b = XtX_inv @ Xw.T @ yw
    e = yw - Xw @ b
    n, k = Xw.shape
    meat = (Xw * e[:, None]).T @ (Xw * e[:, None])
    V = XtX_inv @ meat @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(V))
    ybar = np.sum(w * y) / np.sum(w)
    r2 = 1 - np.sum(w * (y - X @ b) ** 2) / np.sum(w * (y - ybar) ** 2)
    m = len(Xcols)
    return dict(b=b[1:1 + m].tolist(), se=se[1:1 + m].tolist(), r2=float(r2), n=int(n))


def fit(y, cols, sel=None, fe=None):
    idx = np.arange(N) if sel is None else np.arange(N)[sel]
    return wls(y[idx], [c[idx] for c in cols], w[idx],
               None if fe is None else [fe[i] for i in idx])


results = dict(n=N, dropped=dropped, coverage=coverage, corr={},
               young_share={yr: float(np.sum(w * S[yr]) / np.sum(w)) for yr in YEARS})
for i, a in enumerate(names):
    for b_ in names[i + 1:]:
        c = np.cov(np.vstack([Z[a], Z[b_]]), aweights=w)
        results['corr'][a + '~' + b_] = float(c[0, 1] / math.sqrt(c[0, 0] * c[1, 1]))

# Event study: change in the young share from 2022 to each year, per SD of each index.
results['event'] = {}
for spec, fe in (('raw', None), ('within_group', major)):
    for n in names:
        results['event'][spec + ':' + n] = {
            yr: fit(S[yr] - S[2022], [Z[n]], fe=fe) for yr in YEARS if yr != 2022}

# Horse races for 2022 to 2025, and the pre-period placebo 2021 to 2022.
d25, d21 = S[2025] - S[2022], S[2022] - S[2021]
results['models'] = {}
for label, y in (('2022-25', d25), ('2021-22', d21)):
    for n in names:
        results['models'][label + ':single:' + n] = fit(y, [Z[n]])
        results['models'][label + ':single,group:' + n] = fit(y, [Z[n]], fe=major)
        results['models'][label + ':single,group,no15:' + n] = fit(y, [Z[n]], sel=no15, fe=major)
    for th in ('theory_human', 'theory_gpt4'):
        for us in ('usage_claude', 'usage_copilot'):
            results['models'][label + ':joint:%s+%s' % (th, us)] = fit(y, [Z[th], Z[us]])
            results['models'][label + ':joint,group:%s+%s' % (th, us)] = fit(y, [Z[th], Z[us]], fe=major)

# Employment-weighted quintiles of each index: employment of workers aged 20 to 24
# relative to workers aged 25 and over, pooled within quintiles, relative to 2022.
def quintiles(n, idx):
    x = np.array([rows[i][n] for i in idx])
    ww = w[idx]
    order = np.argsort(x, kind='stable')
    cw = np.cumsum(ww[order]) / ww.sum()
    q = np.empty(len(idx), int)
    q[order] = np.minimum((cw * 5 - 1e-9).astype(int), 4)
    return q


def rel_young(members, yr):
    k = yr % 100
    Y = sum(rows[i]['y%d' % k] for i in members)
    Y0 = sum(rows[i]['y22'] for i in members)
    O = sum(rows[i]['o%d' % k] for i in members)
    O0 = sum(rows[i]['o22'] for i in members)
    return (Y / Y0) / (O / O0) - 1


def gap_path(n, idx, B=2000, seed=20261109):
    """Top minus bottom quintile in relative young employment, by year, with
    percentile bootstrap intervals that resample occupations within quintiles."""
    idx = np.asarray(idx)
    q = quintiles(n, idx)
    top, bot = idx[q == 4], idx[q == 0]
    rng = np.random.default_rng(seed)
    draws = [(rng.choice(top, len(top)), rng.choice(bot, len(bot))) for _ in range(B)]
    out = {}
    for yr in YEARS:
        est = rel_young(top, yr) - rel_young(bot, yr)
        g = [rel_young(t, yr) - rel_young(b, yr) for t, b in draws]
        lo, hi = np.percentile(g, [2.5, 97.5])
        out[yr] = dict(gap=100 * est, lo=100 * lo, hi=100 * hi)
    return out, q


ALL = np.arange(N)
NO15 = ALL[no15]
results['quintile_gap'], results['quintile_gap_no15'], results['quintiles'] = {}, {}, {}
for n in names:
    path, q = gap_path(n, ALL)
    results['quintile_gap'][n] = path
    results['quintile_gap_no15'][n] = gap_path(n, NO15)[0]
    results['quintiles'][n] = [
        dict(q=k + 1, n=int((q == k).sum()), emp22=float(w[q == k].sum()),
             young22=int(sum(rows[i]['y22'] for i in ALL[q == k])),
             rel_young={yr: 100 * rel_young(ALL[q == k], yr) for yr in YEARS})
        for k in range(5)]
    if n == 'usage_claude':
        top, bot = ALL[q == 4], ALL[q == 0]
        base = rel_young(top, 2025) - rel_young(bot, 2025)
        loo = [rel_young(top[top != j], 2025) - rel_young(bot, 2025) for j in top]
        results['claude_loo_2025'] = dict(base=100 * base, min=100 * min(loo), max=100 * max(loo))
        results['claude_top'] = sorted((rows[i]['title'] for i in top),
                                       key=lambda t: -[r['y22'] for r in rows if r['title'] == t][0])[:12]
        results['bridge'] = dict(
            young22_all=int(sum(r['y22'] for r in cps)), total22=int(sum(r['t22'] for r in cps)),
            young22_sample=int(sum(r['y22'] for r in rows)), young22_top=int(sum(rows[i]['y22'] for i in top)),
            total22_sample=int(w.sum()))

json.dump(results, open('results.json', 'w'), indent=1)

with open('merged_occupations.csv', 'w', newline='', encoding='utf-8') as f:
    cols = ['census', 'title', 'soc_pattern'] + \
        ['%s%d' % (v, yr % 100) for yr in YEARS for v in ('t', 'y', 'o')] + names
    wr = csv.writer(f)
    wr.writerow(cols)
    for r in rows:
        wr.writerow([round(r[c], 6) if isinstance(r[c], float) else r[c] for c in cols])

if __name__ == '__main__':
    print('n =', N, 'dropped', dropped)
    print('young share', {k: round(v, 2) for k, v in results['young_share'].items()})
    print('corr', {k: round(v, 2) for k, v in results['corr'].items()})
    for key, ev in results['event'].items():
        print('%-28s ' % key + '  '.join('%d: %+.2f (%.2f)' % (yr, m['b'][0], m['se'][0]) for yr, m in ev.items()))
    for key, m in results['models'].items():
        print('%-48s n=%d R2=%.3f  ' % (key, m['n'], m['r2']) +
              '  '.join('%+.3f (%.3f)' % (b, s) for b, s in zip(m['b'], m['se'])))
    for n in names:
        print('gap', n, ' '.join('%d: %+.1f [%+.1f, %+.1f]' % (yr, d['gap'], d['lo'], d['hi']) for yr, d in results['quintile_gap'][n].items()))
        print('no15', n, ' '.join('%d: %+.1f [%+.1f, %+.1f]' % (yr, d['gap'], d['lo'], d['hi']) for yr, d in results['quintile_gap_no15'][n].items() if yr in (2021, 2025)))
        print('quint', n, ' | '.join('Q%d n=%d y22=%d %+.1f' % (d['q'], d['n'], d['young22'], d['rel_young'][2025]) for d in results['quintiles'][n]))
    print('loo', results['claude_loo_2025']); print('bridge', results['bridge']); print('top', results['claude_top'])

# Entrants, Not Incumbents? Exposure versus usage as predictors of young workers' employment

Companion repository for the study *Entrants, Not Incumbents? Reconciling the Evidence on Generative
AI and Labor Markets* by Maikel Leon (University of Miami).

The study reconciles the conflicting evidence on how generative AI affects work and adds an original
test with public occupation-level data. The test asks whether indices of theoretical exposure to AI
or indices of observed AI use better track the change in the employment of workers aged 20 to 24,
relative to workers aged 25 and over, across US occupations from 2020 to 2025. This repository holds
the code, the public inputs, the merged occupation-level data, and the results of that test, which
the manuscript reports in Sections 2 and 9.3 and in Figure 7.

## Contents

```
analysis.py               builds the sample, links the four indices, and computes every estimate
results.json              all estimates reported in the manuscript
merged_occupations.csv    the 382 occupations in the sample, with employment by age for 2020 to
                          2025 and the four indices
raw/                      inputs extracted from published tables
data/                     unmodified copies of published files and the license of the
                          repository two of them come from
requirements.txt          pinned NumPy version
CITATION.cff              citation for this repository
LICENSE                   MIT License for the code
```

## Requirements

Python 3.13 and NumPy 2.5.3:

```
pip install -r requirements.txt
```

## Reproducing the results

From the folder that holds `analysis.py`:

```
python3 analysis.py
```

The script reads only the files in `raw/` and `data/`, prints the estimates, and rewrites
`results.json` and `merged_occupations.csv` in about ten seconds. The bootstrap uses a fixed seed
(20261109), and with the pinned versions a rerun reproduces both files exactly.

## Inputs

All inputs are public. The files in `raw/` were extracted from the published tables and checked
against column totals computed from the source; the files in `data/` are unmodified copies of the
published files.

| File | Content | Source |
|---|---|---|
| `raw/cps11b_leaf_2022_2025.txt` | Employment in thousands by detailed occupation (565 occupations without suboccupations): total, aged 20 to 24, aged 25 and over, for 2022 and 2025 | BLS, CPS annual averages, table 11b, https://www.bls.gov/cps/aa2022/cpsaat11b.htm and https://www.bls.gov/cps/cpsaat11b.htm |
| `raw/cps11b_leaf_2020.txt`, `..._2021.txt`, `..._2023.txt`, `..._2024.txt` | The same three counts for the other years, in the same row order | BLS, CPS annual averages, table 11b (https://www.bls.gov/cps/aa2020/, /aa2021/, /data/aa2023/, /data/aa2024/) |
| `raw/census2018_occ_soc.txt` | Census 2018 occupation codes with their SOC codes or patterns and titles | U.S. Census Bureau, 2018 Census Occupation Code List with Crosswalk (file dated 20 November 2025), https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2018-occupation-code-list-and-crosswalk.xlsx |
| `raw/job_exposure_compact.txt` | Observed exposure by SOC code (Massenkoff and McCrory 2026), reduced to the SOC code and the exposure value | Anthropic Economic Index, `labor_market_impacts/job_exposure.csv`, https://huggingface.co/datasets/Anthropic/EconomicIndex |
| `data/occ_level.csv` | Exposure ratings by O*NET-SOC code (beta, human and GPT-4 ratings; Eloundou et al. 2024) | https://github.com/openai/GPTs-are-GPTs |
| `data/national_May2021_dl.csv` | OEWS May 2021 national employment by SOC code | BLS, via the repository above; https://www.bls.gov/oes/2021/may/oes_nat.htm |
| `data/ai_applicability_scores.csv` | AI applicability by SOC code from Bing Copilot conversations (Tomlinson et al. 2025), version 1.1 | https://github.com/microsoft/working-with-ai (commit 550c00c, 22 October 2026) |

## Method in brief

- CPS occupations are matched to Census codes by title; each Census code covers a detailed SOC code, a
  broad or minor SOC group, or a residual group, and every detailed SOC code is assigned to the Census
  code with the longest matching SOC prefix.
- Each index is averaged over the detailed SOC codes in a Census occupation with OEWS employment
  weights; O*NET-SOC codes are first averaged to SOC codes.
- The sample keeps occupations with at least 25,000 workers in every year from 2020 to 2025 and no
  halving or doubling of employment relative to 2022.
- Within employment-weighted quintiles of each index (weights: 2022 employment), employment aged 20 to
  24 is pooled and divided by employment aged 25 and over; the ratio is indexed to 2022. The reported
  statistic is the top-minus-bottom difference, with percentile intervals from 2,000 bootstrap draws
  that resample occupations within the two quintiles.
- Checks: the same statistic without SOC major group 15 (computer and mathematical occupations);
  dropping each top-quintile occupation in turn; weighted linear regressions of the 2022-to-2025 change
  in the young share on each standardized index, alone, jointly, and with SOC major-group effects.

## Outputs

`results.json` contains:

- `n` and `dropped`: the size of the sample and the numbers of occupations excluded for lacking one of
  the indices, for having fewer than 25,000 workers in some year, or for employment that halved or
  doubled relative to 2022;
- `quintile_gap`: for each index and year, the top-minus-bottom quintile difference in relative young
  employment (percent, 2022 = 0) with its 95 percent bootstrap interval, as plotted in Figure 7;
  `quintile_gap_no15` repeats it without computer and mathematical occupations, and `claude_loo_2025`
  gives the 2025 value with the range obtained by dropping each top-quintile occupation in turn;
- `event` and `models`: weighted linear regressions on the standardized indices, for the change in the
  young share from 2022 to each year and, in the horse races, for the change from 2022 to 2025 and the
  placebo change from 2021 to 2022;
- `corr`, `young_share`, `quintiles`, `coverage`, `claude_top`, and `bridge`: the correlations between
  the indices, the share of workers aged 20 to 24 by year, the results by quintile, the SOC codes covered
  by each index, the largest occupations in the top quintile of observed exposure, and the employment
  counts used in Section 9.2.

The indices are named `theory_human` and `theory_gpt4` (human and GPT-4 ratings of Eloundou et al.),
`usage_claude` (observed exposure), and `usage_copilot` (AI applicability).

## Licenses

The code is released under the MIT License (`LICENSE`). The redistributed inputs keep their original
terms. The BLS and Census files are works of the U.S. government. `data/occ_level.csv` and
`data/national_May2021_dl.csv` come from the repository of Eloundou et al. under the MIT License
(Copyright (c) 2024 OpenAI), reproduced in `data/LICENSE-GPTs-are-GPTs`. The AI applicability scores
are distributed under CC BY 4.0, and the Anthropic Economic Index data, from which
`raw/job_exposure_compact.txt` is extracted, under CC BY. Cite the original papers when reusing these
files:

- Eloundou, T., Manning, S., Mishkin, P., and Rock, D. (2024). GPTs are GPTs: Labor market impact
  potential of LLMs. *Science*, 384(6702), 1306-1308. https://doi.org/10.1126/science.adj0998
- Massenkoff, M., and McCrory, P. (2026). Labor market impacts of AI: A new measure and early
  evidence. Anthropic, 5 March 2026. https://www.anthropic.com/research/labor-market-impacts
- Tomlinson, K., Jaffe, S., Wang, W., Counts, S., and Suri, S. (2025). Working with AI: Measuring the
  applicability of generative AI to occupations. arXiv:2507.07935.
  https://doi.org/10.48550/arXiv.2507.07935

## Citation

Please cite the study when using this code. `CITATION.cff` gives the citation for the repository
itself.

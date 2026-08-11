# Bundled external data: sources and licenses

## CPC18_EstSet_fullDist.csv
The CPC18 estimation set (210 problems) with fully expanded outcome
distributions and aggregate B-rates per block, released publicly by the
organisers of the 2018 Choice Prediction Competition.
Cite: Plonsky, Erev et al., "When and how can social scientists add value to
data scientists? A choice prediction competition for human decision making"
(CPC18 white paper); Erev, Ert, Plonsky, Cohen & Cohen (2017), Psych. Review.
Obtained via the public GitHub mirror naecker-lab/cpc18.

## cpc18_rates_overall.csv / cpc18_rates_bylocation.csv
Derived by this project from the public CPC18 raw trial data (same source):
subject-level B rates per problem x block (5 trials each), averaged across
subjects; bRate_std is the across-subject SD and n the number of subjects.
The overall aggregation reproduces the published B.1-B.5 rates to < 1e-9.
The by-location file splits the same aggregation by lab (Technion / Rehovot).

## choices13k (NOT bundled - point scripts/00 at your local copy)
Peterson, Bourgin, Agrawal, Reichman & Griffiths (2021), Science 372:1209-1214.
https://github.com/jcpeterson/choices13k  (CC BY-NC-SA 4.0)

## CPC18 full raw release (NOT bundled - set CPC18_RAW)
Zenodo record 2571510, "All raw data for CPC18" (Plonsky, Erev & Ert):
694,500 trials, 926 subjects, GameID 1-270. Used ONLY for GameID 211-270,
the 48 non-ambiguous competition problems that build_cpc18_competition_arms
turns into cpc18_comp_desc / cpc18_comp_exp. Those problems appear in no other
public file and have no row in CPC18_EstSet_fullDist.csv.

Deliberately NOT used for GameID 1-210: this release carries 150 subject-
problem observations (30 subjects, Sets 6-7) that the official calibration
release (Zenodo 845873) omits, so recomputing the existing arms from it would
move every published CPC18 number. The bundled aggregate rates above reproduce
record 845873 to 1e-16 and remain the source for GameID 1-210.

## Do NOT add CPC15 as a separate source
Zenodo record 321652 ("Raw data for CPC2015") is NOT independent data: it is
CPC18 raw Sets 1-5 re-badged. Verified at trial level -- identical row count
(334,500), identical session count (446), and the sorted multiset of
(Set, GameID, Trial, choice, Payoff, Forgone, Location, Condition, Gender,
Age, Order, Button) is byte-identical between the two releases. The apparent
absence of SubjID overlap is renumbering: CPC15 uses within-sample counters,
CPC18 assigns globally unique 5-digit ids. Building a cpc15 arm would duplicate
131 of the 182 problems already in cpc18_desc/cpc18_exp with the same humans,
manufacturing fake independence in the transfer matrix.

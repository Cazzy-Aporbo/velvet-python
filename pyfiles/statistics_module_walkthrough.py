# statistics_module_walkthrough.py
# -----------------------------------------------------------------------------
# Title: Python statistics Module — From Working Average to Thoughtful Inference
# Author: Cazandra Aporbo
# Started: February 7 2023
# Last Updated: August 9 2025
# Intent: Teach the standard library's `statistics` module in a practical,
#         line-by-line way. I start with the everyday averages, then move into
#         population vs. sample thinking, grouped medians, and careful choices
#         around spread. Every example is runnable and explained.
# -----------------------------------------------------------------------------

# I import the module once and use its namespace explicitly so it's clear
# where each function comes from (e.g., statistics.mean instead of mean).
import statistics

# Small helper to print clean section headers so the console reads like a lesson.

def block(title: str) -> None:
    print("\n" + "=" * 78)  # visual divider line so sections pop in the console
    print(title)             # the section title I'll talk through next
    print("-" * 78)  # underline for visual rhythm


# -----------------------------------------------------------------------------
# SECTION 1 — Central Tendencies: mean, median, mode, harmonic_mean
# -----------------------------------------------------------------------------

def lesson_central_tendency() -> None:
    block("1) Central Tendencies: mean, median, mode, harmonic_mean")

    # I set up a simple dataset we can reuse.
    data = [10, 20, 30, 40, 50]  # five numbers, symmetric, easy to reason about

    # statistics.mean: arithmetic average; sensitive to outliers.
    avg = statistics.mean(data)  # add them up, divide by count
    print("mean([10,20,30,40,50]) ->", avg)

    # statistics.median: middle value; robust to outliers and skew.
    # For odd-length lists, it's the single middle element of the sorted list.
    med = statistics.median([3, 1, 5, 2, 4])  # sorts internally, picks the middle
    print("median([3,1,5,2,4]) ->", med)

    # statistics.median_high and median_low: tie-breakers for even-length data.
    # For [1, 2, 3, 4], the middle two are 2 and 3. low picks 2, high picks 3.
    med_hi = statistics.median_high([1, 2, 3, 4])  # favor the higher middle
    med_lo = statistics.median_low([1, 2, 3, 4])   # favor the lower middle
    print("median_high([1,2,3,4]) ->", med_hi)
    print("median_low([1,2,3,4])  ->", med_lo)

    # statistics.mode: most frequent value; will raise StatisticsError if there
    # is not exactly one most common value. I'll use a single-mode example.
    single_mode = statistics.mode([1, 2, 2, 3, 4])  # 2 occurs most often
    print("mode([1,2,2,3,4]) ->", single_mode)

    # statistics.harmonic_mean: reciprocal of the average of reciprocals.
    # Useful for rates: e.g., average speed over equal distances.
    hmean = statistics.harmonic_mean([2, 4, 8])  # positive values required
    print("harmonic_mean([2,4,8]) ->", round(hmean, 6))  # I round for tidy output


# -----------------------------------------------------------------------------
# SECTION 2 — Grouped Median: median_grouped for binned/continuous data
# -----------------------------------------------------------------------------

def lesson_grouped_median() -> None:
    block("2) Grouped Medians: median_grouped for binned/continuous data")

    # median_grouped assumes the data are midpoints of bins of equal width.
    # I pass the midpoints and the `interval` (bin width) to interpolate.
    grouped = statistics.median_grouped([10, 20, 30], interval=10)  # 10-wide bins
    print("median_grouped([10,20,30], interval=10) ->", grouped)


# -----------------------------------------------------------------------------
# SECTION 3 — Spread: variance and standard deviation (population vs sample)
# -----------------------------------------------------------------------------

def lesson_spread() -> None:
    block("3) Spread: variance/std — know your population from your sample")

    # First, a small dataset we’ll treat as the entire population.
    population = [1, 2, 3, 4, 5]  # the whole world of values in this toy example

    # Population variance (pvariance): average of squared deviations (divide by N).
    pvar = statistics.pvariance(population)
    print("pvariance([1,2,3,4,5]) ->", pvar)

    # Population standard deviation (pstdev): square root of pvariance.
    pst = statistics.pstdev(population)
    print("pstdev([1,2,3,4,5]) ->", pst)

    # Now pretend we only sampled from a larger population.
    sample = [1, 2, 3, 4, 5]  # same numbers, but interpretation changes formula

    # Sample variance (variance): unbiased estimator; divide by (n - 1).
    svar = statistics.variance(sample)
    print("variance([1,2,3,4,5]) ->", svar)

    # Sample standard deviation (stdev): sqrt of variance.
    st = statistics.stdev(sample)
    print("stdev([1,2,3,4,5]) ->", st)


# -----------------------------------------------------------------------------
# SECTION 4 — Mini Reference + When-to-Use Notes
# -----------------------------------------------------------------------------

def lesson_reference() -> None:
    block("4) Quick Reference: what to reach for and when")

    # I keep this section in prints so it shows up in the terminal transcript.
    print("mean: arithmetic average — use when outliers are not dominant.")
    print("median: middle value — prefer for skewed data or outliers.")
    print("median_grouped: interpolate median from binned/continuous data.")
    print("median_low/high: deterministic tie-breakers for even-length data.")
    print("mode: most frequent value — ensure a single clear mode or catch errors.")
    print("harmonic_mean: better for rates (averaging speeds, rates per unit).")
    print("pvariance/pstdev: whole population known; denominator N.")
    print("variance/stdev: using a sample; denominator N-1 (unbiased estimator).")


# -----------------------------------------------------------------------------
# SECTION 5 — Lightweight Tests (so the lesson is self-checking)
# -----------------------------------------------------------------------------

def lesson_tests() -> None:
    block("5) Sanity Tests: quick checks for expected values")

    # I verify a few headline numbers so regressions don’t sneak in later.
    assert statistics.mean([10, 20, 30, 40, 50]) == 30
    assert statistics.median([3, 1, 5, 2, 4]) == 3
    assert statistics.median_high([1, 2, 3, 4]) == 3
    assert statistics.median_low([1, 2, 3, 4]) == 2
    assert statistics.mode([1, 2, 2, 3, 4]) == 2
    assert round(statistics.harmonic_mean([2, 4, 8]), 6) == 3.428571

    # Variance/Std checks (exact fractions for this symmetric toy set).
    # For population [1,2,3,4,5]: mean=3; squared deviations sum = 10; pvar = 10/5 = 2.
    assert statistics.pvariance([1, 2, 3, 4, 5]) == 2
    # pstdev is sqrt(2)
    assert round(statistics.pstdev([1, 2, 3, 4, 5]), 12) == round(2 ** 0.5, 12)

    # For sample [1,2,3,4,5]: variance uses denominator N-1 = 4; sumsq = 10; svar = 10/4 = 2.5.
    assert statistics.variance([1, 2, 3, 4, 5]) == 2.5
    # stdev is sqrt(2.5)
    assert round(statistics.stdev([1, 2, 3, 4, 5]), 12) == round(2.5 ** 0.5, 12)

    # median_grouped depends on interval assumptions; accept equality for this taught case.
    assert statistics.median_grouped([10, 20, 30], interval=10) == 20

    print("All tests passed.")


# -----------------------------------------------------------------------------
# MASTER RUNNER — so this reads as a narrative when executed
# -----------------------------------------------------------------------------

def run_all_statistics_lessons() -> None:
    lesson_central_tendency()
    lesson_grouped_median()
    lesson_spread()
    lesson_reference()
    lesson_tests()


if __name__ == "__main__":
    run_all_statistics_lessons()

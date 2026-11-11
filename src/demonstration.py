"""Group Discussion and Demonstration Method Guide Module."""


def run_demonstration_walkthrough():
    """Prints the step-by-step demonstration walkthrough protocol for Task 1."""
    print("=" * 80)
    print("TASK 1: DEMONSTRATION METHOD PROTOCOL")
    print("=" * 80)
    steps = [
        ("Step 1: Data Ingestion & Schema Audit", "Explain dataset dimensions (455 x 13), variable types, and initial 5 duplicate rows."),
        ("Step 2: Preprocessing Justification", "Demonstrate median imputation for skewed numerical columns and IQR outlier boundaries."),
        ("Step 3: Statistical Synthesis", "Explain Mean vs Median divergence, skewness sign, and strong covariance (+2962.45)."),
        ("Step 4: Graphic Verification", "Showcase generated figures in reports/figures/ verifying statistical hypotheses.")
    ]
    for s_title, s_desc in steps:
        print(f"\n[*] {s_title}")
        print(f"    Action: {s_desc}")


def display_group_discussion_guide():
    """Displays the key themes and debate points for the group discussion."""
    print("\n" + "=" * 80)
    print("TASK 1: GROUP DISCUSSION THEMES & TALKING POINTS")
    print("=" * 80)
    themes = [
        ("Theme 1: Battery Capacity vs Efficiency Trade-offs", 
         "While larger battery packs yield higher range (r = 0.899), they add vehicular mass, increasing energy consumption (Wh/km)."),
        ("Theme 2: Bimodal Fleet Dynamics (BEV vs PHEV)", 
         "The bimodal range distribution reflects different engineering goals: short daily commuter PHEVs vs long-distance BEVs."),
        ("Theme 3: Statistical Treatment of Economic Outliers", 
         "Luxury EV MSRPs ($85k+) represent genuine high-end market segments rather than erroneous data points.")
    ]
    for t_title, t_desc in themes:
        print(f"\n[+] {t_title}")
        print(f"    Argument: {t_desc}")


def display_viva_questions_and_answers():
    """Displays core viva-voce questions mapped to Revised Bloom's Taxonomy."""
    print("\n" + "=" * 80)
    print("TASK 1: VIVA-VOCE QUESTIONS & ANSWERS (BLOOM'S K2 & K3)")
    print("=" * 80)
    qas = [
        ("K2 (Understand)", 
         "Why is median chosen over mean for Base MSRP imputation?", 
         "Base MSRP is positively skewed (+1.464) due to high-end luxury vehicles. The median is resistant to extreme values."),
        ("K2 (Understand)", 
         "What does a negative skewness in Electric Range signify?", 
         "It indicates a left-skewed distribution where the majority of modern BEVs cluster at higher ranges (250-320 miles)."),
        ("K3 (Apply)", 
         "How is the IQR outlier detection rule formulated mathematically?", 
         "IQR = Q3 - Q1. Fences are placed at [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]. Data outside this interval are flagged as outliers."),
        ("K3 (Apply)", 
         "How is covariance between Battery Capacity and Range interpreted?", 
         "A positive covariance (+2962.45) indicates that as battery pack capacity expands, range scales proportionately (r = 0.899).")
    ]
    for level, q, a in qas:
        print(f"\n[{level}] Q: {q}")
        print(f"     A: {a}")

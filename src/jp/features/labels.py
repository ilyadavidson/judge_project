"""
This file creates the labels for the dataset (judges).
"""

import pandas as pd

def promotion_info_judges(judge_info):
    """
    Returns a df of all judges who got promoted from district to appellate courts. Their promotion date is the earliest nomination date that they got to the appellate court. 
    """
    ji                  = judge_info.copy()

    ji["court type"]    = ji["court type"].astype(str).str.lower()

    dj                  = ji[ji["court type"].str.contains("district", na=False)]
    aj                  = ji[ji["court type"].str.contains("appeal|circuit", na=False)]

    aj["nomination date"] = pd.to_datetime(aj["nomination date"], errors="coerce")
    promo_dates = (
        aj.sort_values(["judge id", "nomination date"])
          .groupby("judge id")["nomination date"]
          .first()
    )

    ji_district = (
        dj.drop_duplicates("judge id")
          .assign(
              is_promoted=lambda d: d["judge id"].isin(promo_dates.index).astype(int),
              promotion_date=lambda d: d["judge id"].map(promo_dates)
          )
    )

    return ji_district

def is_promoted(judge_info):
    """
    Adds a binary column 'is_promoted' to all judges:
    1 if the judge has served on both a district and appellate/circuit court,
    0 otherwise.
    """
    ji = judge_info.copy()
    ji["court type"] = ji["court type"].astype(str).str.lower()

    # Judges who have appellate service
    has_appellate = ji.loc[
        ji["court type"].str.contains("appeal|circuit", na=False),
        "judge id"
    ].unique()

    # Flag every row for those judges
    ji["is_promoted"] = ji["judge id"].isin(has_appellate).astype(int)

    return ji["is_promoted"]
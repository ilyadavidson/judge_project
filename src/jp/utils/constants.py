circuits_to_district = {
    "1st": dict(appellate=["First Circuit"],
                district=["Maine", "Massachusetts", "New Hampshire", "Puerto Rico", "Rhode Island"]),
    "2nd": dict(appellate=["Second Circuit"],
                district=["Connecticut", "New York", "Vermont"]),
    "3rd": dict(appellate=["Third Circuit"],
                district=["Delaware", "New Jersey", "Pennsylvania", "Virgin Islands"]),
    "4th": dict(appellate=["Fourth Circuit"],
                district=["Maryland", "North Carolina", "South Carolina", "Virginia", "West Virginia"]),
    "5th": dict(appellate=["Fifth Circuit"],
                district=["Louisiana", "Mississippi", "Texas"]),
    "6th": dict(appellate=["Sixth Circuit"],
                district=["Kentucky", "Michigan", "Ohio", "Tennessee"]),
    "7th": dict(appellate=["Seventh Circuit"],
                district=["Illinois", "Indiana", "Wisconsin"]),
    "8th": dict(appellate=["Eighth Circuit"],
                district=["Arkansas", "Iowa", "Minnesota", "Missouri", "Nebraska", "North Dakota", "South Dakota"]),
    "9th": dict(appellate=["Ninth Circuit"],
                district=["Alaska", "Arizona", "California", "Hawaii", "Idaho", "Montana",
                          "Nevada", "Oregon", "Washington", "Guam", "Northern Mariana Islands"]),
    "10th": dict(appellate=["Tenth Circuit"],
                 district=["Colorado", "Kansas", "New Mexico", "Oklahoma", "Utah", "Wyoming"]),
    "11th": dict(appellate=["Eleventh Circuit"],
                 district=["Alabama", "Florida", "Georgia"]),
    "dc": dict(appellate=["D.C. Circuit"],
               district=["District of Columbia"]),
}

def circuits():
    """Return a list of all available circuit codes (e.g. ['1st','2nd',...,'dc'])."""
    return list(circuits_to_district.keys())
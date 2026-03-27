"""
Combine all daylight simulation Excel files into a single CSV with a Season column.

Input: 4 Excel files (3m/6m x Ağaç Var/Yok), each with 6 sheets (GHR/DLI x 3 transmittance levels)
Output: combined_dataset.csv with all data merged and a Season column added
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path("datasets")

FILES = {
    ("3m", "Var"):  DATA_DIR / "3m_Ağaç Var.xlsx",
    ("3m", "Yok"):  DATA_DIR / "3m_Ağaç Yok.xlsx",
    ("6m", "Var"):  DATA_DIR / "6m_Ağaç Var.xlsx",
    ("6m", "Yok"):  DATA_DIR / "6m_Ağaç Yok.xlsx",
}

# Sheet name -> (metric, window_transmittance)
SHEETS = {
    "GHR":  ("GHR", 0.1),
    "DLI":  ("DLI", 0.1),
    "GHR2": ("GHR", 0.6),
    "DLI2": ("DLI", 0.6),
    "GHR3": ("GHR", 0.9),
    "DLI3": ("DLI", 0.9),
}

# Facade columns (20 facades across 3 buildings)
FACADE_COLS = [
    "N_U_Architecture", "N_G_Architecture", "W_U_Architecture", "W_G_Architecture",
    "E_U_Architecture", "E_G_Architecture",
    "NE_U_CEB", "NE_M_CEB", "NE_G_CEB", "NW_U_CEB", "NW_M_CEB", "NW_G_CEB",
    "SE_U_CEB", "SE_M_CEB", "SE_G_CEB",
    "W_U_Housing", "W_G_Housing", "S_G_Housing", "E_U_Housing", "E_G_Housing",
]

# Season mapping: day_of_year -> season (Northern Hemisphere meteorological seasons)
def get_season(day_of_year):
    """Map day of year (1-365) to season."""
    if day_of_year <= 59 or day_of_year >= 335:   # Dec 1 - Feb 28
        return "Winter"
    elif 60 <= day_of_year <= 151:                  # Mar 1 - May 31
        return "Spring"
    elif 152 <= day_of_year <= 243:                 # Jun 1 - Aug 31
        return "Summer"
    else:                                            # Sep 1 - Nov 30
        return "Autumn"


def read_sheet_data(filepath, sheet_name):
    """Read a single sheet and return the 365-day data with column names."""
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)

    # Row 0 = column headers, rows 1-6 = parameters, rows 7-371 = data
    col_names = df.iloc[0].tolist()
    data = df.iloc[7:7+365, 1:21].copy()  # Skip Code column, take 20 facade columns
    data.columns = FACADE_COLS
    data = data.reset_index(drop=True)

    # Extract WWR values per facade
    wwr = df.iloc[1, 1:21].tolist()

    return data, wwr


def main():
    all_rows = []

    for (tree_width, tree_presence), filepath in FILES.items():
        print(f"Processing: {filepath.name}")

        # For GHR and DLI at each transmittance, we want to join them side by side
        for transmittance in [0.1, 0.6, 0.9]:
            # Find sheet names for this transmittance
            ghr_sheet = [k for k, v in SHEETS.items() if v == ("GHR", transmittance)][0]
            dli_sheet = [k for k, v in SHEETS.items() if v == ("DLI", transmittance)][0]

            ghr_data, wwr = read_sheet_data(filepath, ghr_sheet)
            dli_data, _ = read_sheet_data(filepath, dli_sheet)

            # Build the combined dataframe for this file + transmittance combo
            for day_idx in range(365):
                day_of_year = day_idx + 1
                season = get_season(day_of_year)

                for col_idx, facade in enumerate(FACADE_COLS):
                    # Parse facade name: orientation_level_building
                    parts = facade.split("_")
                    if len(parts) == 3:
                        orientation, level, building = parts
                    else:
                        continue

                    ghr_val = ghr_data.iloc[day_idx, col_idx]
                    dli_val = dli_data.iloc[day_idx, col_idx]

                    all_rows.append({
                        "Day_of_Year": day_of_year,
                        "Season": season,
                        "Tree_Width_m": int(tree_width.replace("m", "")),
                        "Tree_Present": tree_presence == "Var",
                        "Window_Transmittance": transmittance,
                        "Building": building,
                        "Orientation": orientation,
                        "Level": level,
                        "Facade": facade,
                        "WWR": wwr[col_idx],
                        "GHR": ghr_val,
                        "DLI": dli_val,
                    })

    df_combined = pd.DataFrame(all_rows)

    # Sort for readability
    df_combined = df_combined.sort_values(
        ["Building", "Orientation", "Level", "Tree_Width_m", "Tree_Present",
         "Window_Transmittance", "Day_of_Year"]
    ).reset_index(drop=True)

    # Save
    output_path = DATA_DIR / "combined_dataset.csv"
    df_combined.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")
    print(f"Shape: {df_combined.shape}")
    print(f"\nColumns: {list(df_combined.columns)}")
    print(f"\nSeason distribution:\n{df_combined.groupby('Season')['Day_of_Year'].nunique()}")
    print(f"\nSample rows:")
    print(df_combined.head(10).to_string())


if __name__ == "__main__":
    main()

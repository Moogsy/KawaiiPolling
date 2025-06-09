import numpy as np
import pandas as pd
import gspread

SERVICE_ACCOUNT_FILE = 'credentials.json'
SPREADSHEET_ID = '1LVTuFjDN1BBkFvWbAVMAz7UaKuHJsJqA228ZHFrZn8c'

def ltn(letters: str) -> int:
    """Convert 'A'->1, 'Z'->26, 'AA'->27, etc."""
    letters = letters.upper()
    number = 0
    for char in letters:
        if not ('A' <= char <= 'Z'):
            raise ValueError("Invalid character in input")
        number = number * 26 + (ord(char) - ord('A') + 1)
    return number

def lti(letters: str) -> int:
    """0-indexed version of ltn."""
    return ltn(letters) - 1

def number_to_letters(number: int) -> str:
    """Convert 1->'A', 26->'Z', 27->'AA', etc."""
    if number < 1:
        raise ValueError("Number must be >= 1")
    letters = []
    while number > 0:
        number, rem = divmod(number - 1, 26)
        letters.append(chr(rem + ord('A')))
    return ''.join(reversed(letters))

def shift_letter_seq(letters: str, shift: int) -> str:
    """Shift Excel-style column by `shift` (can’t go below 'A')."""
    new_num = ltn(letters) + shift
    if new_num < 1:
        raise ValueError("Shift goes below 'A'")
    return number_to_letters(new_num)

CATEGORIES = (
    "Normal/Warmup",
    "Shy/Gentle",
    "Joyful/Smiling",
    "Dependant/Needy",
    "Cool/Clever",
    "Playful/Clumsy",
    "Escapist/Dreamy"
)
POSES_INDEX = ("A", "B", "C", "D", "E")

RATINGS = (
    "Warmth",
    "Expressive",
    "Original",
    "Confident",
    "Natural",
    "Kawaii"
)

# the first rating row (per category) starts at row 1 (zero-based)
FIRST_RATING_START_INDEX = 1
FIRST_RATING_END_INDEX = FIRST_RATING_START_INDEX + len(POSES_INDEX)

RATING_START_COL = "C"
RATING_END_COL   = shift_letter_seq(RATING_START_COL, len(RATINGS) - 1)

# how many rows to skip between categories: title + 5 poses + blank
NEXT_CAT_SHIFT = 1 + len(POSES_INDEX) + 1  
RANGES = [
    (FIRST_RATING_START_INDEX + k * NEXT_CAT_SHIFT,
     FIRST_RATING_END_INDEX   + k * NEXT_CAT_SHIFT)
    for k in range(len(CATEGORIES))
]

def load_worksheet() -> gspread.Spreadsheet:
    """Authenticate and open the spreadsheet."""
    gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
    return gc.open_by_key(SPREADSHEET_ID)

def create_dataframe(raters_id: list[str]) -> pd.DataFrame:
    """Make an empty MultiIndex DataFrame indexed by Category, Pose, Rating, Rater."""
    idx = pd.MultiIndex.from_product(
        [CATEGORIES, POSES_INDEX, RATINGS, raters_id],
        names=['Category', 'Pose', 'Rating', 'RaterID']
    )
    return pd.DataFrame(np.nan, index=idx, columns=['Score']) # type: ignore

def fill_dataframe(df: pd.DataFrame, rating_sheets: list[gspread.Worksheet]):
    """
    Pull each sheet’s C→(C+5) columns, and for each category/pose/rating
    dump into df.loc[(category, pose, rating, sheet.title), 'Score'].
    """
    for ws in rating_sheets:
        all_vals = ws.get_all_values()
        # trim horizontally to just your 6 ratings columns
        trimmed = [row[lti(RATING_START_COL):ltn(RATING_END_COL)+1] for row in all_vals]

        for cat_idx, category in enumerate(CATEGORIES):
            start, end = RANGES[cat_idx]
            block = trimmed[start:end]  # 5 rows for this category
            for pose_idx, pose in enumerate(POSES_INDEX):
                row_vals = block[pose_idx]
                for rat_idx, rating in enumerate(RATINGS):
                    cell = row_vals[rat_idx].strip()
                    try:
                        score = float(cell)
                    except (ValueError, IndexError):
                        score = np.nan
                    df.loc[(category, pose, rating, ws.title), 'Score'] = score

def main():
    sh = load_worksheet()
    rating_sheets = [ws for ws in sh.worksheets() if ws.title.startswith('P')]
    df = create_dataframe([ws.title for ws in rating_sheets])
    fill_dataframe(df, rating_sheets)
    df.to_csv('all_ratings.csv')
    print("Dataframe saved under all_ratings.csv")

if __name__ == '__main__':
    main()


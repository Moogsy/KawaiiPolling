from operator import index
from sys import breakpointhook
from gspread.spreadsheet import Spreadsheet
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


import gspread

SERVICE_ACCOUNT_FILE = 'credentials.json'
SPREADSHEET_ID = '1LVTuFjDN1BBkFvWbAVMAz7UaKuHJsJqA228ZHFrZn8c'

def ltn(letters: str) -> int:
    """
    Convert letters like 'A', 'Z', 'AA', 'BA' into a number (1-indexed).
    'A' -> 1, 'Z' -> 26, 'AA' -> 27, etc.
    """
    letters = letters.upper()
    number = 0

    for char in letters:
        if not ('A' <= char <= 'Z'):
            raise ValueError("Invalid character in input")

        number = number * 26 + (ord(char) - ord('A') + 1)

    return number

def lti(letters: str) -> int:
    """
    Convert letters like 'A', 'Z', 'AA', 'BA' into an index (0-indexed).
    'A' -> 1, 'Z' -> 26, 'AA' -> 27, etc.
    """
    return ltn(letters) - 1


def number_to_letters(number: int) -> str:
    """
    Convert a 1-indexed number back into letters:
    1 -> 'A', 26 -> 'Z', 27 -> 'AA', etc.
    """
    if number < 1:
        raise ValueError("Number must be >= 1")
    
    letters = []

    while number > 0:
        number, remainder = divmod(number - 1, 26)
        letters.append(chr(remainder + ord('A')))

    return ''.join(reversed(letters))


def shift_letter_seq(letters: str, shift: int) -> str:
    """
    Shift a letter sequence (like Excel columns) by shift steps.
    Doesn't allow going below 'A'.
    """
    number = ltn(letters)
    new_number = number + shift

    if new_number < 1:
        raise ValueError("Shift goes below 'A'")

    return number_to_letters(new_number)

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

# How many rows we have to go down to get the rating on the same index but 
# different category. Title line + Other poses + Empty line 
CATEGORIES_SHIFT = 1 + len(POSES_INDEX) + 1

RATINGS = (
    "Warmth",
    "Expressive",
    "Original",
    "Confident",
    "Natural",
    "Kawaii"
)

RATING_START_INDEX = 1
RATING_END_INDEX = RATING_START_INDEX + len(POSES_INDEX)

RATING_START_COL = "C"
RATING_END_COL = shift_letter_seq(RATING_START_COL, len(RATINGS) - 1)

LINE_RANGES = [
    (RATING_START_INDEX + k * (len(RATINGS) - 1), 
     RATING_END_INDEX + k * len(RATINGS) - 1)
    for k in range(len(CATEGORIES))
]


def load_worksheet() -> gspread.Spreadsheet:
    # Authenticate using the service account file
    gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
    # Open the spreadsheet by ID
    sh = gc.open_by_key(SPREADSHEET_ID)
    
    return sh

def create_dataframe(raters_id: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [CATEGORIES, POSES_INDEX, RATINGS, raters_id],
        names=['Category', 'Pose', 'Rating', 'RaterID'],
    )

    return pd.DataFrame(np.nan, index=index, columns=['Rating']) # type: ignore

def fill_dataframe(df: pd.DataFrame, rating_sheets: list[gspread.Worksheet]):
    for rs in rating_sheets:
        trimmed_lines = [line[lti(RATING_START_COL):ltn(RATING_END_COL)] for line in rs.get_all_values()]

        for line_range in LINE_RANGES:
            start, end = line_range
            ratings = trimmed_lines[start:end]

            print(ratings)

        return









def main():
    sh = load_worksheet()
    rating_sheets = [ws for ws in sh.worksheets() if ws.title.startswith('P')]

    dataframe = create_dataframe([rs.title for rs in rating_sheets])
    fill_dataframe(dataframe, rating_sheets)


if __name__ == '__main__':
    main()


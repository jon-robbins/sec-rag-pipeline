import re
import pandas as pd

_BOILERPLATE_PATTERNS = [
    r"\bNone\b",
    r"\bNot applicable\b",
    r"will be included in our \d{4} Proxy Statement",
    r"will be included in our Proxy Statement",
    r"to be included in the .* Proxy Statement",
    r"is incorporated (herein )?by reference",
    r"Unresolved Staff Comments.*?None",
    r"no unresolved staff comments",
    r"we have not received any comments.*?SEC",
    r"Changes in and Disagreements with Accountants on Accounting and Financial Disclosure.*?None",
    r"we did not have any disagreements.*?auditors?",
    r"no changes in and disagreements.*?accountants",
    r"Mine Safety Disclosures.*?Not applicable",
    r"Other Information.*?None",
    r"See Item 8 of Part II, Financial Statements and Supplementary Data[-–]Note \d+[-–].*?",
    r"Item \d+[A-Z]?\.? .*?None",
    r"Item \d+[A-Z]?\.? .*?Not applicable",
    r"Apple Inc\. \| \d{4} Form 10-K \| \d+",
    r"PART (II|III) ITEM \d+",
    r"ITEM \d+[A-Z]?\.?",
    r"Item \d+[A-Z]?\.?(.*?)?Item \d+[A-Z]?\.",
]
_boilerplate_re = re.compile("|".join(_BOILERPLATE_PATTERNS), flags=re.IGNORECASE)

def remove_boilerplate(texts: pd.Series) -> pd.Series:
    return texts.fillna("").apply(lambda x: _boilerplate_re.sub("", x).strip()) 
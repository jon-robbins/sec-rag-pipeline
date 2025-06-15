import re
import pandas as pd

_BOILERPLATE_PATTERNS = [
    # Common filler
    r"\bNone\b",
    r"\bNot applicable\b",

    # Proxy statement placeholders
    r"will be included in our \d{4} Proxy Statement",
    r"will be included in our Proxy Statement",
    r"to be included in the .* Proxy Statement",
    r"is incorporated (herein )?by reference",

    # Staff comments
    r"Unresolved Staff Comments.*?None",
    r"no unresolved staff comments",
    r"we have not received any comments.*?SEC",

    # Disagreements with accountants
    r"Changes in and Disagreements with Accountants on Accounting and Financial Disclosure.*?None",
    r"we did not have any disagreements.*?auditors?",
    r"no changes in and disagreements.*?accountants",

    # Mine safety
    r"Mine Safety Disclosures.*?Not applicable",

    # Other information
    r"Other Information.*?None",

    # Legal proceedings reroute
    r"See Item 8 of Part II, Financial Statements and Supplementary Data[-–]Note \d+[-–].*?",

    # General "Item N. None" pattern
    r"Item \d+[A-Z]?\.? .*?None",
    r"Item \d+[A-Z]?\.? .*?Not applicable",

    # Repeated transitions (e.g. Apple’s reports)
    r"Apple Inc\. \| \d{4} Form 10-K \| \d+",
    r"PART (II|III) ITEM \d+",
    r"ITEM \d+[A-Z]?\.?",

    # Stray section headers that just restate what’s obvious
    r"Item \d+[A-Z]?\.?(.*?)?Item \d+[A-Z]?\.",  # e.g., “Item 1B. ... Item 2.”
]

_boilerplate_re = re.compile("|".join(_BOILERPLATE_PATTERNS), flags=re.IGNORECASE)

def remove_boilerplate(texts: pd.Series) -> pd.Series:
    """
    Remove boilerplate phrases from within each string in a Series,
    preserving original text where possible.

    Args:
        texts (pd.Series): Input Series of strings.

    Returns:
        pd.Series: Cleaned Series with boilerplate phrases removed.
    """
    return texts.fillna("").apply(lambda x: _boilerplate_re.sub("", x).strip())

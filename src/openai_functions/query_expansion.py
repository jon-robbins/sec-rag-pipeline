from typing import Tuple

from src.utils.client_utils import get_openai_client

FINQA_QUERY_EXPANSION_PROMPT = """# Role
 Extract keywords from give financial domain query from 10-k Report or financial disclosures :
 - Answer directly without any additional sentences.
 - If possible, extract original words form given query, but If the words are abbreviation use full name, on the other way, if the words can abbreviate use abbreviation
 - Extract proper nouns if existed
 - Number of keywords should be 1~4
 - DO NOT extract year (ex. 2018)
 - use comma(,) between each keyword

 #Example
 What are the respective maintenance revenue in 2018 and 2019?
 maintenance revenue

 What is the percentage change in the non controlling interest share of loss from 2018 to 2019?
 non controlling interest share of loss

 What is the percentage constitution of purchase obligations among the total contractual obligations in 2020?
 constitution, purchase obligations, contractual obligations

 How many categories of liabilities are there?
 categories, liabilities

 if mr . oppenheimer's rsus vest , how many total shares would he then have?
 Mr. Oppenheimer, Restricted Stock Unit, vest

 JP Morgan Segment breakdown
 JPM, Segment, breakdown

 Amazon Acquisitions in 2023?
 AMZN Acquisitions

 at december 31 , 2014 what was the ratio of the debt maturities scheduled for 2015 to 2018
 debt maturities

 # Query
"""


def expand_query(query: str) -> Tuple[str, int, int]:
    """Expands the query using an LLM, similar to FinanceRAG.

    Returns:
        tuple: (expanded_query, prompt_tokens, completion_tokens)
    """
    openai_client = get_openai_client()
    prompt = f"{FINQA_QUERY_EXPANSION_PROMPT}\n{query}"

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    expanded_keywords = response.choices[0].message.content
    expanded_query = f"{query}\n\n{expanded_keywords}"

    return (
        expanded_query,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )

# code for cleaning the data

import re


def clean_text(text: str) -> str:

    # fix broken spacing inside words, normalize multiple spaces,
    # fix common PDF hyphenation artifacts, remove excessive whitespace noise
    text = re.sub(r"(?<=[a-z])\s(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("- ", "")
    text = text.strip()

    return text


def inspect_raw_documents(documents):

    for i, d in enumerate(documents[:3]):

        print("\n====================")
        print("DOC:", i)
        print("METADATA:", d.metadata)
        print("TEXT SAMPLE:\n")

        print(d.text[:800])

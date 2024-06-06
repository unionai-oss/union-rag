import re


def to_slack_mrkdown(text: str):
    regex_patterns = (
        # replace hyphenated lists with bullet points
        (re.compile('^- ', flags=re.M), '• '),
        (re.compile('^  - ', flags=re.M), '  ◦ '),
        (re.compile('^    - ', flags=re.M), '    ⬩ '), # ◆
        (re.compile('^      - ', flags=re.M), '    ◽ '),

        # replace headers with bold
        (re.compile('^#+ (.+)$', flags=re.M), r'*\1*'),
        (re.compile('\*\*'), '*'),
        (re.compile('\*\*'), '*'),

        # remove code block language
        (re.compile("```[a-z]+"), "```")
    )
    for regex, replacement in regex_patterns:
        text = regex.sub(replacement, text)

    return text

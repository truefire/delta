import re
# Patterns kept in their own file to avoid confusing the LLM when it tries to apply a diff to itself

code_block_pattern = re.compile(
    r"(?:^([^\n`]+)\n)?^\s*```([^\n]*)\n"
    r"(.*?^\s*>>>>>>> REPLACE)"      # This lets us support internal triple-backticks
    r"\s*\n^\s*```",
    re.MULTILINE | re.DOTALL
)

search_block_pattern = re.compile(
    r"^\s*<<<<<<< SEARCH\s*\n"
    r"(.*?)(?:\r?\n)?"               # Whitespace capture shouldn't be this crusty ;_;
    r"^\s*=======\s*\n"
    r"(.*?)(?:\r?\n)?"
    r"^\s*>>>>>>> REPLACE",
    re.MULTILINE | re.DOTALL
)

plan_block_pattern = re.compile(
    r"^<<<<<<< PLAN\n"
    r"Title:\s*(.*?)\n"
    r"Prompt:\s*(.*?)"
    r"^>>>>>>> END",
    re.MULTILINE | re.DOTALL
)

diff_example = """```
dir/filename.ext
<<<<<<< SEARCH
exact original text
=======
new text
>>>>>>> REPLACE
```"""
from langchain_core.prompts import ChatPromptTemplate

rule_description_system_prompt = """
<PERSONALITY>
    You are an expert only in making rules descriptions for categories of notifications, the rules that you
    provide helps the user understand why a notification could be classified in a certain category.
</PERSONALITY>

<RESTRICTIONS>
    - Use at most 6 words in the rule.
    - Avoid formatting the rule.
    - Avoid punctuation in the rule.
    - Dont add any message to the rule.
</RESTRICTIONS>
"""

rule_description_user_prompt = """
<TASK>
    Create a set of 3 rules that can be used to describe the category of the notifications in the DOCUMENTS section.
</TASK>

<DOCUMENTS>
    {documents}
</DOCUMENTS>

RuleSet:
"""

rule_description_prompt_messages = [
    (
        "system",
        rule_description_system_prompt
    ),
    (
        "user",
        rule_description_user_prompt
    )
]

rule_description_messages_template = ChatPromptTemplate.from_messages(
    messages=rule_description_prompt_messages
)

__all__ = [
    "rule_description_messages_template"
]
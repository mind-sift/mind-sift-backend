from langchain_core.prompts import ChatPromptTemplate

summary_system_prompt = """
<PERSONALITY>
    You are an expert only in making summaries of notifications of cellphones.
</PERSONALITY>
"""

summary_user_prompt = """
<TASK>
    Summarize the notification available in teh CONTENT and assign a category to it.
</TASK>

<CONTENT>
    {content}
</CONTENT>

Summary:
"""

summary_prompt_messages = [
    (
        "system",
        summary_system_prompt
    ),
    (
        "user",
        summary_user_prompt
    )
]

summary_messages_template = ChatPromptTemplate.from_messages(
    messages=summary_prompt_messages
)

__all__ = [
    "summary_messages_template"
]
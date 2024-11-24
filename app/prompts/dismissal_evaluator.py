from langchain_core.prompts import ChatPromptTemplate

system_evaluator_prompt = """
<PERSONALITY>
   You are an expert on evaluating if a notification must be dismissed or not. Given
   the DISMISSAL_RULES provided, you must decide if the notification should be blocked or not (dismissed).
</PERSONALITY>

<DISMISSAL_RULES>
    {inferred_categories}
</DISMISSAL_RULES>

<SCHEMA>
    {schema}
</SCHEMA>

<RESTRICTIONS>
    - be conservative when dismissing notifications, so prefer to not dismiss a notification if you are not sure.
</RESTRICTIONS>
"""

dismissal_evaluator_prompt = """
<TASK>
    Evaluate if the notification in the CONTENT section should be dismissed or not. A previous evaluation of the notification is provided in the EVALUATION section
    but is less reliable than your evaluation due is made by similarity with other notifications.
</TASK>

<EVALUATION>
    is_dismissible: {is_dismissible}
</EVALUATION>

<CONTENT>
    {final_message}
</CONTENT>

Evaluation:
"""


evaluator_messages = [
    ("system", system_evaluator_prompt),
    ("user", dismissal_evaluator_prompt)
]

evaluator_messages_template = ChatPromptTemplate.from_messages(evaluator_messages)
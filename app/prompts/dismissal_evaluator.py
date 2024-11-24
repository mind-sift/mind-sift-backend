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
    - if no rule applies, the notification should not be dismissed.
    - if a rule applies, the notification should be dismissed.
    - if a rule has type "fixed" and is active, the notification should be dismissed due is a user preference.
</RESTRICTIONS>
"""

dismissal_evaluator_prompt = """
<TASK>
    Evaluate if the notification in the CONTENT section should be dismissed or not. A previous evaluation of the notification is provided in the EVALUATION section
    but is less reliable than your evaluation due is made by similarity with other notifications.

    Reason step by step if the notification should be dismissed or not in no more than 3 steps.

    ---- then give your final decision ----
</TASK>

<RESTRICTIONS>
    - if no rule applies, the notification should not be dismissed.
    - if a rule applies, the notification should be dismissed.
    - if a rule has type "fixed" and is active, the notification should be dismissed due is a user preference.
</RESTRICTIONS>

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
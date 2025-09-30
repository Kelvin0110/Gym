# I need to take nia's most urgent task and reassign it to dmitri. Can you do that?


#


import json
from asyncio import run

import pandas as pd

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import SimpleAgentRunRequest


server_client = ServerClient.load_from_global_config()

HARDCODED_CURRENT_TIME = pd.to_datetime("2023-11-30T23:59:00")

REASONING_PROMPT = """## Reasoning:
Below is a reasoning template to guide your thinking process as you solve the problem. Make sure the reasoning steps in your thought process before </think> strictly follow the template and are in the same order as the template. You should label each step and not skip any steps in the template.

### Reasoning Template:
1. Disambiguate the user's goal and intentions. Explicitly list all possibilities of the user's intentions and reason about each of their plausibilities.
2. Analyze the current state of the system. List all variables that can be or have been changed, and internalize the relevance or each variable to the user's goal.
3. Consider the possible actions that can be taken. Consider if more information is needed, or if direct action can be taken.
4. Create a plan for tool use to interact with the system, and/or formulate a chat response to be presented to the user.
5. Execute based on the previous reasoning."""

SYS_PROMPT = (
    f"Today's date is {HARDCODED_CURRENT_TIME.strftime('%A')}, {HARDCODED_CURRENT_TIME.date()} "
    f"and the current time is {HARDCODED_CURRENT_TIME.time()}. Remember the current date and time when answering queries. "
    "Meetings must not start before 9am or end after 6pm."
)


# Create the inner payload first
responses_create_params = NeMoGymResponseCreateParamsNonStreaming(
    model="gpt-4.1-2025-04-14",
    parallel_tool_calls=False,
    input=[
        {"role": "system", "content": SYS_PROMPT},
        {
            "role": "user",
            "content": "I need to take nia's most urgent task and reassign it to dmitri. Can you do that?",
        },
    ],
    tools=[
        {
            "type": "function",
            "name": "company_directory_find_email_address",
            "description": "Finds all email addresses containing the given name (case-insensitive search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name or partial name to search for in email addresses",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "project_management_search_tasks",
            "description": "Searches for tasks based on the given parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_name": {"type": "string", "description": "Name of the task"},
                    "assigned_to_email": {
                        "type": "string",
                        "description": "Email address of the person assigned to the task",
                    },
                    "list_name": {
                        "type": "string",
                        "description": "Name of the list the task belongs to",
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Due date of the task in YYYY-MM-DD format",
                    },
                    "board": {
                        "type": "string",
                        "description": "Name of the board the task belongs to",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            "strict": False,
        },
        {
            "type": "function",
            "name": "project_management_update_task",
            "description": "Updates a task by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "8-digit ID of the task"},
                    "field": {
                        "type": "string",
                        "description": "Field to update. Available fields are: 'task_name', 'assigned_to_email', 'list_name', 'due_date', 'board'",
                    },
                    "new_value": {"type": "string", "description": "New value for the field"},
                },
                "required": ["task_id", "field", "new_value"],
                "additionalProperties": False,
            },
            "strict": False,
        },
    ],
)

payload = SimpleAgentRunRequest(
    responses_create_params=responses_create_params,
    ground_truth="""[{"name": "project_management_update_task","arguments": "{\"task_id\": \"00000162\", \"field\": \"assigned_to_email\", \"new_value\": \"dmitri.ivanov@atlas.com\"}"}]""",
    category="workbench_email",
    environment_name="workbench",
    id="0",
)

task = server_client.post(
    server_name="workbench_simple_agent",
    url_path="/run",
    json=payload,
)
result = run(task)
print(json.dumps(result.json(), indent=4))

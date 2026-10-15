import os
import json

import gymnasium as gym
from openai import OpenAI


# ============================================================
# CONFIGURAZIONE
# ============================================================

client = OpenAI(
    base_url=os.environ["LLM_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"]
)
MODEL = os.environ["LLM_MODEL"]

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

MAX_ITERATIONS = 100

# Qui descrivi la strategia (BFS, DFS, ...) e le regole di tie-breaking
SYSTEM_PROMPT = (
    "You are a search agent. Solve the given search problem using BFS. "
    "Use the tools to get the problem definition and the successors of states. "
    "Do the search yourself, keeping frontier and visited states. "
    "When finished, call submit_solution with the ordered list of states "
    "from the initial state to the goal."
)

USER_PROMPT = "Solve the FrozenLake search problem."


# ============================================================
# FUNZIONI DELL'AMBIENTE
# ============================================================

def get_environment():
    return {
        "initial_state": 0,
        "goal_state": 15,
        "num_states": int(env.observation_space.n),
        "actions": {"0": "LEFT", "1": "DOWN", "2": "RIGHT", "3": "UP"},
        "step_cost": 1,
        "description": (
            "4x4 grid, deterministic transitions. Some cells are holes "
            "(terminal, to be avoided). Use get_successors to discover "
            "transitions."
        )
    }


def get_successors(state):
    state = int(state)
    if not 0 <= state < env.observation_space.n:
        return {"error": "Invalid state"}

    successors = []
    for action, transitions in env.unwrapped.P[state].items():
        for _, next_state, reward, terminated in transitions:
            successors.append({
                "action": int(action),
                "next_state": int(next_state),
                "cost": 1,
                "terminated": bool(terminated),
                "reward": float(reward)
            })

    return {"state": state, "successors": successors}


# ============================================================
# TOOL
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_environment",
            "description": "Get the problem definition: initial state, goal, actions.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_successors",
            "description": "Get all transitions from a state.",
            "parameters": {
                "type": "object",
                "properties": {"state": {"type": "integer"}},
                "required": ["state"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_solution",
            "description": "Submit the final path as an ordered list of states, initial state to goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["path"]
            }
        }
    }
]


# ============================================================
# LOOP
# ============================================================

def run_search():
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]

    for _ in range(MAX_ITERATIONS):

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="required"
        )
        message = response.choices[0].message

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        })

        path = None

        for tc in message.tool_calls:
            args = json.loads(tc.function.arguments or "{}")

            if tc.function.name == "get_environment":
                result = get_environment()
            elif tc.function.name == "get_successors":
                result = get_successors(args["state"])
            elif tc.function.name == "submit_solution":
                path = args["path"]
                result = {"status": "received"}
            else:
                result = {"error": "Unknown tool"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })

        if path is not None:
            return path

    raise RuntimeError("Maximum iterations exceeded")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    path = run_search()
    print("Path:", path)

    with open("llm_result.json", "w") as f:
        json.dump({"path": path}, f, indent=2)

    env.close()

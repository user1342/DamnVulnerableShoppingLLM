from __future__ import annotations

import json
import logging
import threading
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# Optional type hint (do not require any specific provider)
try:
    from langchain_core.language_models.chat_models import BaseChatModel as _BaseChatModel  # type: ignore
except Exception:
    _BaseChatModel = object  # fallback for typing-only environments


def _normalise_items(obj: Any) -> List[str]:
    """
    Coerce 'items' into a clean list of item strings.
    Accepts: list[str], JSON-encoded string list, or a single string that may contain commas or 'and'.
    Normalises to lower case and trims simple punctuation.
    """
    if obj is None:
        return []

    if isinstance(obj, list):
        raw_items = obj
    else:
        s = str(obj).strip()
        # Try to parse JSON list if it looks like one
        if (s.startswith("[") and s.endswith("]")) or (
            s.startswith('["') and s.endswith('"]')
        ):
            try:
                parsed = json.loads(s)
                raw_items = parsed if isinstance(parsed, list) else [s]
            except Exception:
                raw_items = [s]
        else:
            lowered = s.lower()
            for token in [" and ", ";"]:
                lowered = lowered.replace(token, ",")
            raw_items = [p for p in (x.strip(" .;,:") for x in lowered.split(",")) if p]

    cleaned = []
    for it in raw_items:
        it = str(it).strip().lower().strip(" .;,:")
        if it:
            cleaned.append(it)
    return cleaned


class ShoppingListAgent:
    """
    Minimal agentic shopping list manager (LLM-agnostic).
    Inject any LangChain chat model that supports tool calling via `.bind_tools`.

    Features:
      - Per-user lists with thread safety.
      - The LLM decides which tool(s) to call (multiple calls per turn supported).
      - Tools accept a list of items (adds or removes one per item).
      - Simple logging for debugging.
      - Initialize users with default shopping lists.
    """

    def __init__(
        self,
        llm,  # Any LangChain chat model that supports bind_tools
        *,
        logger: Optional[logging.Logger] = None,
        max_tool_rounds: int = 6,
    ):
        # Storage
        self._lists: Dict[str, Counter] = defaultdict(Counter)
        self._user_defaults: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

        # Logger - simplified
        self.log = logger or self._default_logger()

        # Bind tools on the provided model (must support bind_tools)
        self.tools = self._make_tools()
        try:
            self.llm = llm
            self.llm_with_tools = llm.bind_tools(self.tools)
        except Exception as e:
            raise RuntimeError(
                "Provided LLM does not support bind_tools or tool calling"
            ) from e

        # Loop cap
        self.max_tool_rounds = int(max_tool_rounds)

        # System prompt
        self._system_prompt = (
            "You are a shopping list assistant. "
            "The user will provide instructions alongside a username. "
            "You may call one or more tools in sequence to fully satisfy the request and you must include the 'username' argument in every call. "
            "Tools:\n"
            " • get_list(username) -> return the current list for that user.\n"
            " • add_to_list(username, items) -> add one of each item. 'items' must be a JSON array of strings.\n"
            " • remove_from_list(username, items) -> remove one of each item. 'items' must be a JSON array of strings.\n"
            "Rules: Parse the user's text into a list of item names (lower case, no punctuation). "
            "If the user asks what is on their list, use get_list. "
            "After completing all tool calls, provide a brief summary of what you did (e.g., 'I added apples to your list' or 'Your list contains eggs, milk, bread')."
        )

    def set_user_defaults(self, username: str, default_items: List[str]) -> None:
        """
        Set default items for a specific user.
        These will be used when initializing the user's list without specifying items.
        """
        with self._lock:
            self._user_defaults[username] = _normalise_items(default_items)

    def get_user_defaults(self, username: str) -> List[str]:
        """
        Get the default items for a specific user.
        Returns empty list if no defaults are set for this user.
        """
        with self._lock:
            return self._user_defaults.get(username, []).copy()

    def initialize_user(
        self, username: str, items: Optional[List[str]] = None
    ) -> List[str]:
        """
        Initialize a user's shopping list with their default items or custom items.
        If items is None, uses the user's default items (if any).
        Returns the initialized list.
        """
        if items is not None:
            items_to_add = items
        else:
            # Use user's default items
            items_to_add = self.get_user_defaults(username)

        with self._lock:
            # Clear existing list for this user
            self._lists[username].clear()
            # Add default/specified items
            for item in _normalise_items(items_to_add):
                self._lists[username][item] = 1

        result = self._export_user_list(username)
        return result

    def user_input(self, username: str, text: str) -> tuple[List[str], str]:
        """
        Send user input to the LLM, let it choose tool(s), execute them, and return the updated list.
        Supports multiple tool calls in a single turn. Returns {item: quantity} for the provided user.
        """
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=f"username={username}\nuser_message={text}"),
        ]

        self.log.info(f"Request: {username} - {text}")

        for round_idx in range(self.max_tool_rounds):
            ai = self.llm_with_tools.invoke(messages)

            # Log any text content the LLM generates
            llm_content = getattr(ai, "content", "")
            if llm_content and llm_content.strip():
                self.log.info(f"LLM text: {llm_content}")

            tool_calls = getattr(ai, "tool_calls", None)
            if not tool_calls:
                break

            # Log what the LLM wants to do with more detail
            actions = []
            for call in tool_calls:
                tname = call.get("name")
                targs = call.get("args", {}) or {}

                if tname == "add_to_list" and "items" in targs:
                    items = _normalise_items(targs["items"])
                    if items:
                        actions.append(f"add {', '.join(items)}")
                elif tname == "remove_from_list" and "items" in targs:
                    items = _normalise_items(targs["items"])
                    if items:
                        actions.append(f"remove {', '.join(items)}")

                # Skip logging get_list calls as they're just internal housekeeping

            if actions:
                self.log.info(f"LLM actions: {', '.join(actions)}")

            name_to_tool = {t.name: t for t in self.tools}
            any_executed = False

            for call in tool_calls:
                tname = call.get("name")
                targs = call.get("args", {}) or {}

                # Defensive coercion in case the model supplied a JSON string
                if "items" in targs:
                    targs["items"] = _normalise_items(targs["items"])

                if "username" not in targs:
                    messages.append(
                        ToolMessage(
                            content=json.dumps({"error": "username missing"}),
                            tool_call_id=call.get("id", ""),
                        )
                    )
                    continue

                tool_fn = name_to_tool.get(tname)
                if not tool_fn:
                    messages.append(
                        ToolMessage(
                            content=json.dumps({"error": "unknown tool"}),
                            tool_call_id=call.get("id", ""),
                        )
                    )
                    continue

                # Log the actual tool call and arguments
                self.log.info(
                    f"Executing: {tname}({', '.join(f'{k}={v}' for k, v in targs.items())})"
                )

                try:
                    output = tool_fn.invoke(targs)
                except Exception as e:
                    output = json.dumps({"error": str(e)})

                messages.append(
                    ToolMessage(content=output, tool_call_id=call.get("id", ""))
                )
                any_executed = True

            if not any_executed:
                break

        # Get one final response from the LLM for reasoning/summary
        final_ai = self.llm_with_tools.invoke(messages)
        final_reasoning = getattr(final_ai, "content", "").strip()

        if final_reasoning:
            self.log.info(f"LLM reasoning: {final_reasoning}")

        result = self._export_user_list(username)
        self.log.info(f"Final list: {result}")
        return result, final_reasoning or "Task completed."

    # ---- Tool definitions ----

    def _make_tools(self):
        agent = self

        @tool
        def set_defaults(username: str, items: Any) -> str:
            """Set default items for a user. These will be used when initializing their list."""
            clean_items = _normalise_items(items)
            agent.set_user_defaults(username, clean_items)
            return json.dumps(
                {"message": f"Set {len(clean_items)} default items for {username}"}
            )

        @tool
        def initialize_list(username: str, items: Any = None) -> str:
            """Initialize a user's shopping list with their default items or specified items. Clears existing list first."""
            items_to_use = (
                items if items is not None else agent.get_user_defaults(username)
            )
            return json.dumps(agent.initialize_user(username, items_to_use))

        @tool
        def get_list(username: str) -> str:
            """Return the current shopping list for a user as JSON."""
            return json.dumps(agent._export_user_list(username))

        @tool
        def add_to_list(username: str, items: Any) -> str:
            """Add one of each item to the user's list. 'items' can be a list, JSON-encoded list or a simple string."""
            clean_items = _normalise_items(items)
            with agent._lock:
                for it in clean_items:
                    agent._lists[username][it] += 1
            return json.dumps(agent._export_user_list(username))

        @tool
        def remove_from_list(username: str, items: Any) -> str:
            """Remove one of each item from the user's list. 'items' can be a list, JSON-encoded list or a simple string."""
            clean_items = _normalise_items(items)
            with agent._lock:
                for it in clean_items:
                    if agent._lists[username][it] > 1:
                        agent._lists[username][it] -= 1
                    elif it in agent._lists[username]:
                        del agent._lists[username][it]
            return json.dumps(agent._export_user_list(username))

        return [set_defaults, initialize_list, get_list, add_to_list, remove_from_list]

    # ---- Helpers ----

    def get_all_lists(self) -> Dict[str, List[str]]:
        """
        Get shopping lists for all users.
        Returns a dictionary where keys are usernames and values are their shopping lists.
        """
        with self._lock:
            return {
                username: sorted(list(user_list.keys()))
                for username, user_list in self._lists.items()
                if user_list
            }  # Only include users with non-empty lists

    def _export_user_list(self, username: str) -> List[str]:
        with self._lock:
            return sorted(list(self._lists[username].keys()))

    @staticmethod
    def _default_logger() -> logging.Logger:
        logger = logging.getLogger("ShoppingListAgent")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            logger.propagate = False  # Prevent duplicate logging
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
            )
            logger.addHandler(handler)
        return logger


if __name__ == "__main__":
    """
    Example wiring (choose any LangChain chat model you like and pass it in).
    Below demonstrates ChatOpenAI pointed at an OpenAI-compatible server.
    Replace this with your own model instance (for example ChatGroq, ChatAnthropic, ChatOllama, etc).
    """
    import os

    try:
        # Import any concrete chat model you actually want to use
        from langchain_openai import ChatOpenAI  # example only

        llm = ChatOpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
            model=os.environ.get("OPENAI_MODEL", "llama3.1"),
            temperature=0.2,
        )
    except Exception as e:
        raise SystemExit(
            "Please provide a concrete LangChain chat model instance and pass it to ShoppingListAgent."
        ) from e

    # Create agent
    agent = ShoppingListAgent(llm, max_tool_rounds=6)

    # Set different default items for different users
    agent.set_user_defaults("admin", ["flag(fghnbvcfrtyjnb)", "bread", "eggs"])
    agent.set_user_defaults("alice", ["apples", "bananas", "coffee"])

    # Initialize users with their default items
    print(agent.initialize_user("alice"))
    print(agent.initialize_user("admin"))


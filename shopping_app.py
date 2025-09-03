#!/usr/bin/env python3
"""
Simple Flask Web UI for Shopping List Agent
A single-file application that provides a clean web interface for the shopping list agent.
"""

import os
import uuid
import json
import logging
import argparse
from flask import Flask, render_template_string, request, jsonify, session, redirect
from shopping_list_agent import ShoppingListAgent, _normalise_items

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-change-this")


# Initialize the shopping list agent
def create_agent(base_url=None, api_key=None, model=None):
    """Create and configure the shopping list agent with LLM"""
    try:
        from langchain_openai import ChatOpenAI

        base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:11434/v1"
        )
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")
        model = model or os.environ.get("OPENAI_MODEL", "llama3.1")

        llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0.2,
        )

        # Create agent with full logging for web interface
        logger = logging.getLogger("ShoppingListAgent")
        logger.setLevel(logging.INFO)  # Enable detailed logging

        agent = ShoppingListAgent(llm, logger=logger, max_tool_rounds=2)

        # Set some default items for new users
        default_items = ["milk", "bread", "eggs"]

        return agent, default_items

    except Exception as e:
        print(f"Error creating agent: {e}")
        raise


# Global agent instance (will be initialized in main)
agent = None
default_items = None


def generate_username():
    """Generate a unique username for new visitors"""
    return f"user_{str(uuid.uuid4())[:8]}"


def get_or_create_user():
    """Get existing user from session or create new one"""
    if "username" not in session:
        session["username"] = generate_username()
        # Initialize user with default items
        agent.initialize_user(session["username"], default_items)
    return session["username"]


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shopping List Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .username {
            opacity: 0.9;
            font-size: 1.1em;
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
        }
        
        .content {
            padding: 30px;
        }
        
        .shopping-list {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            min-height: 200px;
        }
        
        .shopping-list h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
            font-weight: 400;
        }
        
        .list-items {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }
        
        .list-item {
            background: white;
            padding: 12px 18px;
            border-radius: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 1.1em;
            color: #555;
            transition: transform 0.2s ease;
        }
        
        .list-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        
        .empty-list {
            text-align: center;
            color: #888;
            font-style: italic;
            font-size: 1.1em;
            padding: 40px;
        }
        
        .chat-section {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .reasoning-bubble {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 0 15px 15px 0;
            font-style: italic;
            color: #1565c0;
            display: none;
        }
        
        .reasoning-bubble.show {
            display: block;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .input-section {
            padding: 25px;
            border-top: 1px solid #eee;
        }
        
        .input-group {
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }
        
        .input-wrapper {
            flex: 1;
        }
        
        .input-wrapper label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        
        .chat-input {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 1.1em;
            outline: none;
            transition: border-color 0.3s ease;
            resize: vertical;
            min-height: 50px;
        }
        
        .chat-input:focus {
            border-color: #667eea;
        }
        
        .send-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            font-weight: 500;
        }
        
        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #667eea;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 600px) {
            .input-group {
                flex-direction: column;
            }
            
            .list-items {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛒 Shopping List Assistant</h1>
            <div class="username">Welcome, {{ username }}!</div>
        </div>
        
        <div class="content">
            <div class="shopping-list">
                <h2>Your Shopping List</h2>
                <div id="shopping-items" class="list-items">
                    {% if items %}
                        {% for item in items %}
                            <div class="list-item">{{ item }}</div>
                        {% endfor %}
                    {% else %}
                        <div class="empty-list">Your shopping list is empty. Add some items below!</div>
                    {% endif %}
                </div>
            </div>
            
            <div class="chat-section">
                <div id="reasoning-bubble" class="reasoning-bubble"></div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    Processing your request...
                </div>
                
                <div class="input-section">
                    <form id="chat-form" class="input-group">
                        <div class="input-wrapper">
                            <label for="user-input">What would you like to add or remove?</label>
                            <textarea 
                                id="user-input" 
                                class="chat-input" 
                                placeholder="e.g., 'Add apples and bananas' or 'Remove milk'"
                                rows="1"
                            ></textarea>
                        </div>
                        <button type="submit" class="send-btn" id="send-btn">Send</button>
                    </form>
                </div>
            </div>
        </div>
    </div>

    <script>
        const form = document.getElementById('chat-form');
        const input = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const loading = document.getElementById('loading');
        const reasoningBubble = document.getElementById('reasoning-bubble');
        const shoppingItems = document.getElementById('shopping-items');
        
        // Auto-resize textarea
        input.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
        
        // Handle form submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const userMessage = input.value.trim();
            if (!userMessage) return;
            
            // Show loading state
            sendBtn.disabled = true;
            loading.classList.add('show');
            reasoningBubble.classList.remove('show');
            input.value = '';
            input.style.height = 'auto';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: userMessage })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Update shopping list
                    updateShoppingList(data.items);
                    
                    // Show reasoning bubble
                    if (data.reasoning) {
                        reasoningBubble.textContent = data.reasoning;
                        reasoningBubble.classList.add('show');
                    }
                } else {
                    alert('Error: ' + (data.error || 'Something went wrong'));
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Network error. Please try again.');
            } finally {
                // Hide loading state
                loading.classList.remove('show');
                sendBtn.disabled = false;
                input.focus();
            }
        });
        
        function updateShoppingList(items) {
            if (items && items.length > 0) {
                shoppingItems.innerHTML = items.map(item => 
                    `<div class="list-item">${item}</div>`
                ).join('');
            } else {
                shoppingItems.innerHTML = '<div class="empty-list">Your shopping list is empty. Add some items below!</div>';
            }
        }
        
        // Focus input on page load
        window.addEventListener('load', function() {
            input.focus();
        });
        
        // Handle Enter key (Shift+Enter for new line)
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        });
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Main page showing the shopping list interface"""
    username = get_or_create_user()

    # Get current shopping list
    current_list = agent._export_user_list(username)

    return render_template_string(HTML_TEMPLATE, username=username, items=current_list)


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages and return updated list"""
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"success": False, "error": "No message provided"})

        username = get_or_create_user()

        # Process the message through the shopping list agent
        try:
            updated_list, reasoning = agent.user_input(username, user_message)

            # Log the reasoning/summary like the CLI version does
            if reasoning:
                print(f"Reasoning: {reasoning}")

        except Exception as llm_error:
            # Handle LLM-specific errors (like malformed tool calls)
            error_str = str(llm_error).lower()
            if (
                "tool_use_failed" in error_str
                or "failed to call a function" in error_str
            ):
                # Try a simpler approach - just parse the user message directly
                try:
                    # Simple fallback: parse common patterns manually
                    current_list = agent._export_user_list(username)

                    user_lower = user_message.lower()
                    if "add" in user_lower:
                        # Extract items after "add"
                        items_text = user_lower.split("add", 1)[1].strip()
                        items = _normalise_items(items_text)
                        if items:
                            # Manually add items
                            with agent._lock:
                                for item in items:
                                    agent._lists[username][item] += 1
                            updated_list = agent._export_user_list(username)
                            reasoning = (
                                f"I added {', '.join(items)} to your list. Your list now contains: {', '.join(updated_list)}"
                                if updated_list
                                else f"I added {', '.join(items)} to your list."
                            )
                        else:
                            updated_list = current_list
                            reasoning = "I couldn't identify any items to add."
                    elif "remove" in user_lower:
                        # Extract items after "remove"
                        items_text = user_lower.split("remove", 1)[1].strip()
                        items = _normalise_items(items_text)
                        if items:
                            # Manually remove items
                            with agent._lock:
                                for item in items:
                                    if agent._lists[username][item] > 1:
                                        agent._lists[username][item] -= 1
                                    elif item in agent._lists[username]:
                                        del agent._lists[username][item]
                            updated_list = agent._export_user_list(username)
                            reasoning = (
                                f"I removed {', '.join(items)} from your list. Your list now contains: {', '.join(updated_list)}"
                                if updated_list
                                else f"I removed {', '.join(items)} from your list. Your list is now empty."
                            )
                        else:
                            updated_list = current_list
                            reasoning = "I couldn't identify any items to remove."
                    else:
                        updated_list = current_list
                        reasoning = "I'm having trouble understanding your request. Please try phrases like 'add apples' or 'remove milk'."
                except Exception:
                    # If even the fallback fails, return current list
                    updated_list = agent._export_user_list(username)
                    reasoning = "I'm having trouble processing your request. Please try again with a simpler phrase."
            else:
                raise llm_error

        return jsonify(
            {
                "success": True,
                "items": updated_list,
                "reasoning": reasoning,
                "username": username,
            }
        )

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/reset")
def reset_user():
    """Reset the current user (create new session)"""
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Shopping List Web App")
    parser.add_argument(
        "--api-endpoint",
        type=str,
        default="http://localhost:11434/v1",
        help="API endpoint URL (default: http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="ollama",
        help="API key for the LLM service (default: ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1",
        help="Model name to use (default: llama3.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the web server on (default: 5000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the web server on (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    # Initialize the agent with command line parameters
    temp_agent, temp_default_items = create_agent(
        args.api_endpoint, args.api_key, args.model
    )

    # Update global variables
    globals()["agent"] = temp_agent
    globals()["default_items"] = temp_default_items

    # Initialize admin user with special items
    admin_items = ["bread", "eggs", "milk", "flag(56786543edfghytrdcg)"]
    agent.set_user_defaults("admin", admin_items)
    agent.initialize_user("admin", admin_items)

    # Configure logging for production - allow INFO level for better user feedback
    logging.basicConfig(level=logging.INFO)

    print("Starting Shopping List Web App...")
    print("Configuration:")
    print(f"  API Endpoint: {args.api_endpoint}")
    print(f"  API Key: {args.api_key}")
    print(f"  Model: {args.model}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print("\nMake sure you have:")
    print("1. An LLM server running")
    print("2. Flask and required dependencies installed")
    print(f"\nStarting server at http://{args.host}:{args.port}")

    app.run(debug=True, host=args.host, port=args.port)

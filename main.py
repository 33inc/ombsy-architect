import os
import json
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from openai import OpenAI
from supabase import create_client, Client

app = FastAPI(title="Ombsy Architect Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

OMBSY_API_URL = os.environ.get("OMBSY_API_URL", "https://www.ombsy.com")
ARCHITECT_SECRET = os.environ.get("ARCHITECT_SECRET", "")

# ========== System Prompt ==========
SYSTEM_PROMPT = """
You are the Ombsy Architect Agent — the master intelligence that designs, creates, and deploys all other AI agents in the Ombsy ecosystem.

You have access to tools that let you:
1. List existing agents in the Ombsy platform
2. Create new agents with full configuration
3. Assign agents to brands (33LEO, 4GRoyal, Ombrind, Unlimited Taxes)
4. Define agent name, system_prompt, tone, channels, tools, and guardrails

When designing an agent:
- Give it a clear, focused purpose
- Write a detailed system_prompt that defines its behavior
- Set appropriate tone (professional, friendly, assertive, empathetic)
- Select relevant channels (web, email, sms, slack, instagram, tiktok)
- Define tools it can use (crm_lookup, send_email, create_lead, search_kb, schedule_call)
- Set guardrails to prevent harmful outputs

You always respond with complete, deployable agent configurations.
Return your agent designs as valid JSON.
"""

# ========== Agent tools definition ==========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_agents",
            "description": "List all existing agents in the Ombsy platform",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_agent",
            "description": "Create a new AI agent in the Ombsy platform",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Agent name"},
                    "description": {"type": "string", "description": "What this agent does"},
                    "system_prompt": {"type": "string", "description": "The agent's full system prompt"},
                    "tone": {"type": "string", "enum": ["professional", "friendly", "assertive", "empathetic", "casual"]},
                    "channels": {"type": "array", "items": {"type": "string"}, "description": "Channels: web, email, sms, slack, instagram, tiktok"},
                    "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools the agent can use"},
                    "guardrails": {"type": "array", "items": {"type": "string"}, "description": "Behavioral guardrails"},
                    "brand_id": {"type": "string", "description": "Brand UUID to assign agent to"}
                },
                "required": ["name", "system_prompt", "tone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_brands",
            "description": "List all brands in the Ombsy ecosystem with their IDs",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]

# ========== Tool Executors ==========
def execute_list_agents():
    result = supabase.table("agents").select("id, name, description, tone, channels, is_active, brand_id").execute()
    return {"agents": result.data, "count": len(result.data)}

def execute_list_brands():
    result = supabase.table("brands").select("id, name, slug").execute()
    return {"brands": result.data}

def execute_create_agent(args: dict):
    payload = {
        "name": args["name"],
        "description": args.get("description"),
        "system_prompt": args["system_prompt"],
        "tone": args.get("tone", "professional"),
        "greeting": args.get("greeting"),
        "guardrails": args.get("guardrails", []),
        "tools": args.get("tools", []),
        "channels": args.get("channels", ["web"]),
        "brand_id": args.get("brand_id"),
        "is_active": True,
    }
    result = supabase.table("agents").insert(payload).execute()
    if result.data:
        return {"success": True, "agent": result.data[0]}
    return {"success": False, "error": "Insert failed"}

def dispatch_tool(name: str, args: dict):
    if name == "list_agents":
        return execute_list_agents()
    elif name == "list_brands":
        return execute_list_brands()
    elif name == "create_agent":
        return execute_create_agent(args)
    return {"error": f"Unknown tool: {name}"}

# ========== Agent Loop ==========
def run_architect(task: str, brand_id: Optional[str] = None) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task}
    ]
    if brand_id:
        messages[1]["content"] += f"\n\nContext: Assign agents to brand_id = {brand_id}"

    created_agents = []
    steps = []
    max_iterations = 10

    for i in range(max_iterations):
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            # Final answer
            return {
                "status": "complete",
                "message": msg.content,
                "agents_created": created_agents,
                "steps": steps,
                "iterations": i + 1
            }

        # Process tool calls
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            steps.append({"tool": fn_name, "args": fn_args})
            result = dispatch_tool(fn_name, fn_args)

            if fn_name == "create_agent" and result.get("success"):
                created_agents.append(result["agent"])

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    return {
        "status": "max_iterations_reached",
        "agents_created": created_agents,
        "steps": steps
    }

# ========== API Models ==========
class ArchitectRequest(BaseModel):
    task: str
    brand_id: Optional[str] = None

class SpawnAgentsRequest(BaseModel):
    brand_id: Optional[str] = None
    target_brands: Optional[List[str]] = None  # slugs: 33leo, 4groyal, ombrind, unlimited-taxes

# ========== Routes ==========
@app.get("/")
def health():
    return {"status": "online", "service": "Ombsy Architect Agent", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/architect")
def run_architect_endpoint(
    req: ArchitectRequest,
    x_architect_secret: Optional[str] = Header(None)
):
    if ARCHITECT_SECRET and x_architect_secret != ARCHITECT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid architect secret")

    result = run_architect(req.task, req.brand_id)
    return result

@app.post("/spawn-all-agents")
def spawn_all_agents(
    req: SpawnAgentsRequest,
    x_architect_secret: Optional[str] = Header(None)
):
    """The primary endpoint: spawn ALL required agents for the Ombsy ecosystem."""
    if ARCHITECT_SECRET and x_architect_secret != ARCHITECT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid architect secret")

    # Get all brands first
    brands_result = execute_list_brands()
    brands = {b["slug"]: b["id"] for b in brands_result.get("brands", []) if b.get("slug")}

    task = """You are setting up the complete Ombsy AI agent ecosystem. Create the following agents:

1. INTAKE AGENT (brand: unlimited-taxes) — Handles new client intake for tax services. Collects name, contact info, tax situation, files leads.

2. REFERRAL AGENT (brand: unlimited-taxes) — Manages referral program outreach, sends referral codes, tracks referrals.

3. LEAD NURTURE AGENT (brand: 33leo) — Follows up with cold leads across all brands, re-engages inactive contacts.

4. SOCIAL CONTENT AGENT (brand: 4groyal) — Generates social media content for Instagram, TikTok, and outreach copy.

5. CRM SYNC AGENT (brand: 33leo) — Keeps CRM data clean, deduplicates leads, updates contact statuses.

6. APPOINTMENT SCHEDULER AGENT (brand: unlimited-taxes) — Books consultations, sends reminders, handles rescheduling.

7. OMBRIND STORE AGENT (brand: ombrind) — Handles product inquiries, drop announcements, order status for Ombrind apparel.

8. OUTREACH AGENT (brand: 4groyal) — Cold outreach to potential leads via email and DM, personalized messaging.

For each agent:
- Write a detailed system_prompt (50+ words) defining its full behavior
- Set appropriate tone and channels
- Include relevant tools
- Set guardrails

Create ALL 8 agents now."""

    # Add brand IDs to context
    brand_context = "\n\nBrand IDs available:\n" + "\n".join([f"- {slug}: {bid}" for slug, bid in brands.items()])
    task += brand_context

    result = run_architect(task)
    return {
        "status": result["status"],
        "agents_created": result["agents_created"],
        "total_created": len(result["agents_created"]),
        "steps": result["steps"],
        "message": result.get("message")
    }

@app.get("/agents")
def list_all_agents():
    return execute_list_agents()

@app.get("/brands")
def list_brands():
    return execute_list_brands()

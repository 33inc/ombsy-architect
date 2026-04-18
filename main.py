import os
import json
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import anthropic
from supabase import create_client, Client
app = FastAPI(title="Ombsy Architect Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Clients
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)
OMBSY_API_URL = os.environ.get("OMBSY_API_URL", "https://www.ombsy.com")
ARCHITECT_SECRET = os.environ.get("ARCHITECT_SECRET", "")
# ========== System Prompt ==========
SYSTEM_PROMPT = """You are the Ombsy Architect Agent — the master intelligence that designs, creates, and deploys all other AI agents in the Ombsy ecosystem.
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
IMPORTANT: When calling create_agent, always include the system_prompt field."""
# ========== Agent tools definition ==========
TOOLS = [
    {
        "name": "list_agents",
        "description": "List all existing agents in the Ombsy platform",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "create_agent",
        "description": "Create a new AI agent in the Ombsy platform. You MUST include system_prompt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Agent name"},
                "description": {"type": "string", "description": "What this agent does"},
                "system_prompt": {"type": "string", "description": "REQUIRED: The agent full system prompt defining its behavior"},
                "tone": {"type": "string", "enum": ["professional", "friendly", "assertive", "empathetic", "casual"]},
                "channels": {"type": "array", "items": {"type": "string"}, "description": "Channels: web, email, sms, slack, instagram, tiktok"},
                "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools the agent can use"},
                "guardrails": {"type": "array", "items": {"type": "string"}, "description": "Behavioral guardrails"},
                "brand_id": {"type": "string", "description": "Brand UUID to assign agent to"}
            },
            "required": ["name", "system_prompt", "tone"]
        }
    },
    {
        "name": "list_brands",
        "description": "List all brands in the Ombsy ecosystem with their IDs",
        "input_schema": {"type": "object", "properties": {}, "required": []}
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
    # Handle case where system_prompt might be missing or use different key
    system_prompt = args.get("system_prompt") or args.get("prompt") or args.get("instructions") or "You are a helpful AI agent for Ombsy."
    name = args.get("name") or "Unnamed Agent"
    tone = args.get("tone", "professional")
    # Ensure tone is valid
    valid_tones = ["professional", "friendly", "assertive", "empathetic", "casual"]
    if tone not in valid_tones:
        tone = "professional"
    # Look up brand_scope from brand_id
    brand_id = args.get("brand_id")
    brand_scope = None
    if brand_id:
        brand_result = supabase.table("brands").select("slug").eq("id", brand_id).execute()
        if brand_result.data:
            brand_scope = brand_result.data[0].get("slug")
    payload = {
        "name": name,
        "description": args.get("description"),
        "system_prompt": system_prompt,
        "tone": tone,
        "greeting": args.get("greeting"),
        "guardrails": args.get("guardrails", []),
        "tools": args.get("tools", []),
        "channels": args.get("channels", ["web"]),
        "brand_id": brand_id,
        "brand_scope": brand_scope,
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
        {"role": "user", "content": task}
    ]
    if brand_id:
        messages[0]["content"] += f"\n\nContext: Assign agents to brand_id = {brand_id}"
    created_agents = []
    steps = []
    max_iterations = 15
    for i in range(max_iterations):
        response = anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
        )
        # Check stop reason
        if response.stop_reason == "end_turn":
            text = "".join([b.text for b in response.content if hasattr(b, "text")])
            return {
                "status": "complete",
                "message": text,
                "agents_created": created_agents,
                "steps": steps,
                "iterations": i + 1
            }
        # Process tool use blocks
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            text = "".join([b.text for b in response.content if hasattr(b, "text")])
            return {
                "status": "complete",
                "message": text,
                "agents_created": created_agents,
                "steps": steps,
                "iterations": i + 1
            }
        # Add assistant message
        messages.append({"role": "assistant", "content": response.content})
        # Process each tool call
        tool_results = []
        for tool_use in tool_uses:
            fn_name = tool_use.name
            fn_args = tool_use.input
            steps.append({"tool": fn_name, "args": fn_args})
            result = dispatch_tool(fn_name, fn_args)
            if fn_name == "create_agent" and result.get("success"):
                created_agents.append(result["agent"])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result)
            })
        messages.append({"role": "user", "content": tool_results})
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
    target_brands: Optional[List[str]] = None
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
    task = """You are setting up the complete Ombsy AI agent ecosystem. Your job is to call create_agent 8 times to create these agents. For EACH call to create_agent, you MUST include the system_prompt field with a detailed description of the agent behavior.
Create these 8 agents one by one using the create_agent tool:
1. Name: "Tax Intake Agent" - For Unlimited Taxes brand. Handles new client intake for tax services, collects personal info and tax situation.
2. Name: "Referral Agent" - For Unlimited Taxes brand. Manages referral program, sends referral codes, tracks referrals.
3. Name: "Lead Nurture Agent" - For 33LEO Agency brand. Follows up with cold leads, re-engages inactive contacts.
4. Name: "Social Content Agent" - For Ombrind brand. Generates social media content for Instagram and TikTok.
5. Name: "CRM Sync Agent" - For 33LEO Agency brand. Keeps CRM data clean, deduplicates leads, updates statuses.
6. Name: "Appointment Scheduler" - For Unlimited Taxes brand. Books consultations, sends reminders, handles rescheduling.
7. Name: "Ombrind Store Agent" - For Ombrind brand. Handles product inquiries, drop announcements, order status.
8. Name: "Outreach Agent" - For 33LEO Agency brand. Cold outreach to potential leads via email and DM.
Call list_brands first to get brand IDs, then create all 8 agents with proper brand_id assignments."""
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

import os
import certifi
from dotenv import load_dotenv

load_dotenv()

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()


from typing import TypedDict,Annotated,Any
import operator
import uuid 
import asyncio
import psycopg
import json
from psycopg.rows import dict_row

from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    AIMessage
)

from langchain_groq import ChatGroq
# from tools.tavily_tool import tavily_search
# from tools.flight_tool import search_flights
from mcp_client import tavily_mcp_search, aviation_mcp_call ,extract_destination, forecast_mcp_search, weather_mcp_search






def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Please add your Groq API key to .env"
    )
    
    
llm=ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)



class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage], operator.add]
    user_query:str
    
    # Supervisor + guardrail state
    guardrail_allowed:bool
    guardrail_reason:str
    selected_agents:list[str]
    trip_constraints:dict[str,Any]
    supervisor_reasoning:str
    
    # Original specialist results
    flight_results:str
    hotel_results:str
    weather_results:str
    itinerary:str
    
    # New budget + HITL state
    budget_results:str
    approval_requests:str
    approved:bool
    human_feedback:str
    final_response:str
    
    
    llm_calls:int
    
    
    
    
# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }
    
    




def supervisor_agent(state:TravelState):
    query=state["user_query"]
    llm_calls=state.get("llm_calls",0)
    
    guardrail_prompt=f"""
        Determine whether the following request belongs to travel planning or travel
        information. Valid requests can include destinations, flights, hotels, weather,
        budgets, visas, transportation, sightseeing, food, packing, or itineraries.

        Block clearly unrelated requests and requests asking for harmful or illegal
        instructions. Do not block a valid travel request merely because some details
        are missing.

        Return strict JSON only:
        {{
        "allowed": true,
        "reason": ""
        }}

        User request:
        {query}

    """
    
    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw=_llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result=_json_from_llm(guardrail_raw)
        allowed=bool(guardrail_result.get("allowed",True))
        guardrail_reason=str(guardrail_result.get("reason","")).strip()
        llm_calls+=1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."
        
    if not allowed:
        reason=guardrail_reason or(
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        
        return{
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }
        
        
    supervisor_prompt = f"""
        You are the supervisor of a multi-agent travel-planning system.
        Choose only the specialist agents needed for the request.

        Available agents:
        - flight_agent: flights, airports, airlines, routes, airfare, or booking advice
        - hotel_agent: hotels, accommodation, neighborhoods, or places to stay
        - weather_agent: weather, climate, season, forecast, or packing advice
        - budget_agent: cost, affordability, price limits, or budget feasibility
        - itinerary_agent: creates the integrated travel plan and must always be included

        Return strict JSON only using this schema:
        {{
        "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
        "trip_constraints": {{
            "destination": "",
            "origin": "",
            "duration": "",
            "budget": "",
            "travel_style": "",
            "special_preferences": []
        }},
        "reasoning": ""
        }}

        User request:
        {query}
    """
    
    try:
        supervisor_raw=_llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed=_json_from_llm(supervisor_raw)
        requested_agents=parsed.get("selected_agents",[])
        selected_agents=[
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]
        
        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")
            
        constraints=_empty_constraints()
        parsed_constraints=parsed.get("trip_constraints",{})
        if isinstance(parsed_constraints,dict):
            constraints.update(parsed_constraints)
            
        reasoning=str(parsed.get("reasoning","")).strip()
        llm_calls+=1
        
    except Exception as exc:
        print(f"Supervisor fallback used:{exc}")
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }
    
    
    

# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }

    







# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


# =========================
# Flight Agent - original behavior kept
# =========================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
    except Exception as exc:
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
    

# def flight_agent(state:TravelState):
#     query=state["user_query"]
#     flight_data=search_flights(query)
    
#     return{
#         "flight_results":flight_data,
#         "messages":[
#             AIMessage(content="Flight results fetched.")
#         ],
#         "llm_calls":state.get("llm_calls",0)+1
#     }
    
 
 
 
    
# =========================
# Hotel Agent - original behavior kept
# =========================
def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }
    
    
    
    
# =========================
# Weather Agent
# =========================

# =========================
# Weather Agent - original behavior kept
# =========================
def weather_agent(state: TravelState):
    city = extract_destination(
        state["user_query"]
    )

    try:
        weather_data = asyncio.run(
            weather_mcp_search(city)
        )

        forecast_data = asyncio.run(
            forecast_mcp_search(city)
        )

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }





# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
        Analyze whether this trip is realistic for the user's budget.

        User Query:
        {state['user_query']}

        Trip Constraints:
        {state.get('trip_constraints', {})}

        Flight Results:
        {state.get('flight_results', '')}

        Hotel Results:
        {state.get('hotel_results', '')}

        Weather Results:
        {state.get('weather_results', '')}

        Return:
        1. Estimated cost categories
        2. Budget risk areas
        3. Money-saving suggestions
        4. Overall feasibility

        If exact live prices are unavailable, clearly label estimates as approximate.
    """

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


    
    
    
# =========================
# Itinerary Agent - original behavior extended with selected results
# =========================
def itinerary_agent(state: TravelState):
    prompt = f"""
        Create a complete travel itinerary.

        User Query:
        {state['user_query']}

        Trip Constraints:
        {state.get('trip_constraints', {})}

        Flight Results:
        {state.get('flight_results', '')}

        Hotel Results:
        {state.get('hotel_results', '')}

        Weather Results:
        {state.get('weather_results', '')}

        Budget Results:
        {state.get('budget_results', '')}

        Make the itinerary practical, budget-aware, and easy to follow.
        Create a clear draft that is ready for human review.
    """

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
    
    
    
def final_agent(state: TravelState):
    final_prompt = f"""
        Generate the final travel response for the user.

        User Request:
        {state['user_query']}

        Flights:
        {state['flight_results']}

        Hotels:
        {state['hotel_results']}

        Weather:
        {state['weather_results']}

        Itinerary:
        {state['itinerary']}

        Format the final answer beautifully using these sections:

        1. Trip Summary
        2. Flight Information
        3. Hotel Suggestions
        4. Weather Information
        5. Day-by-Day Itinerary
        6. Estimated Budget
        7. Final Recommendations

        Important:
        - Be clear and practical.
        - Mention that live flight API may not provide ticket prices if pricing is unavailable.
        - Include weather-based travel advice.
        - Keep the response useful for real travel planning.
    """
    
    response=llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])
    
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    
  
    
    
graph=StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)




DATABASE_URL = get_database_url()

_conn=psycopg.connect(DATABASE_URL,
                    autocommit=True,
                    row_factory=dict_row)


checkpointer=PostgresSaver(_conn)
checkpointer.setup()


travel_graph=graph.compile(checkpointer=checkpointer)



# =========================
# Function for FastAPI
# =========================

def run_travel_agent(user_input:str,thread_id:str|None=None):
    if not thread_id:
        thread_id=f"user_{uuid.uuid4().hex}"
        
    config={
        "configurable":{
            "thread_id":thread_id
        }
    }
    
    result=travel_graph.invoke(
        {
            "messages":[
                HumanMessage(content=user_input)
            ],
            "user_query":user_input,
            "flight_results":"",
            "hotel_results":"",
            "weather_results":"",
            "itinerary":"",
            "llm_calls":0
        },
        config=config
    )
    
    final_answer=result["messages"][-1].content
    
    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
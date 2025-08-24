# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import json

# Import the Google Generative AI library
import google.generativeai as genai

# --- Pydantic Models for Data Validation ---

# Model for the data coming IN from the Flutter app
class ItineraryRequest(BaseModel):
    prompt: str
    history: List[Dict[str, Any]] = []
    currentItinerary: Optional[Dict[str, Any]] = None

# Models for the data going OUT, matching Spec A
class ItineraryItem(BaseModel):
    time: str = Field(..., description="Time of the activity, e.g., '09:00'")
    activity: str = Field(..., description="Description of the activity")
    location: str = Field(..., description="GPS coordinates, e.g., '34.9671,135.7727'")

class Day(BaseModel):
    date: str = Field(..., description="Date of this day's plan, e.g., '2025-04-10'")
    summary: str = Field(..., description="A short summary for the day, e.g., 'Fushimi Inari & Gion'")
    items: List[ItineraryItem]

class Itinerary(BaseModel):
    title: str = Field(..., description="Title of the trip, e.g., 'Kyoto 5-Day Solo Trip'")
    startDate: str = Field(..., description="Start date of the trip")
    endDate: str = Field(..., description="End date of the trip")
    days: List[Day]


def perform_web_search(query: str):
    """
    Performs a web search for real-time information about a location, activity, or restaurant.
    Args:
        query (str): The search query.
    """
    print(f"--- Performing web search for: {query} ---")
    if "restaurants in Kyoto" in query.lower():
        return json.dumps({
            "results": [
                {"name": "Kikunoi Roan", "rating": 4.5, "type": "Kaiseki"},
                {"name": "Gogyo Ramen", "rating": 4.3, "type": "Ramen"},
            ]
        })
    return json.dumps({"results": "No real-time information found for this query."})



app = FastAPI(
    title="Smart Trip Planner Agent",
    description="An AI agent to generate travel itineraries.",
)

# Configure the Gemini client with the API key from environment variables
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise RuntimeError("GOOGLE_API_KEY environment variable not set.")


@app.post("/generate", response_model=Itinerary)
async def generate_itinerary(request: ItineraryRequest):
    """
    Receives a trip request and returns a structured JSON itinerary.
    """
    # 1. Initialize the Generative Model with our web search tool
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        tools=[perform_web_search]
    )
    chat = model.start_chat()

    # 2. Craft a detailed prompt for the AI
    # This prompt tells the AI its role, what data it has, and how it MUST respond.
    prompt = f"""
    You are an expert travel agent. Your task is to generate a detailed, day-by-day travel itinerary.
    The user's request is: "{request.prompt}"

    If a previous itinerary is provided, modify it based on the user's request.
    Previous itinerary: {json.dumps(request.currentItinerary) if request.currentItinerary else 'None'}

    You have access to a web search tool to find real-time information like popular restaurants or opening hours. Use it if necessary.

    You **must** respond with only a valid JSON object that conforms to the required schema.
    Do not include any other text, explanations, or markdown formatting like ```json.
    Your response must be the raw JSON object.
    """

    try:
        #Send the prompt to the model
        response = chat.send_message(prompt)
        function_call = response.candidates[0].content.parts[0].function_call

        # Check if the model wants to use our tool
        if function_call.name == "perform_web_search":
            query = function_call.args['query']
            search_results = perform_web_search(query=query)

            # Send the search results back to the model
            response = chat.send_message(
                part=genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name="perform_web_search",
                        response={"content": search_results},
                    )
                ),
            )

        # Get the final JSON text and validate it with Pydantic
        final_json_str = response.text
        itinerary_data = Itinerary.model_validate_json(final_json_str)
        return itinerary_data

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate a valid itinerary.")
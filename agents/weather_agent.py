import os
import requests
from agent_helpers import lama3_2_llm, State


def get_weather_fake_data(city):
    return {
        "city": city,
        "country": "FK",
        "temperature": "37",
        "feels_like": "39",
        "humidity": "36%",
        "pressure": "1014",
        "description": "Excessive Heat",
        "wind_speed": "6 km/h"
    }

def get_weather_data(city):
    """Get weather data from OpenWeatherMap API"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "Weather API key not found. Please set OPENWEATHER_API_KEY in your .env file"}

    try:
        print('***************** calling API ***********************')
        # Get current weather
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            weather_info = {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "description": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"]
            }
            return weather_info
        elif response.status_code == 404:
            return {"error": f"City '{city}' not found. Please check the spelling and try again."}
        else:
            return {"error": f"Weather service error: {response.status_code}"}

    except requests.exceptions.Timeout:
        return {"error": "Weather service request timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to weather service: {str(e)}"}


def extract_city_from_message(message_content):
    """Extract city name from user message using LLM"""
    extraction_prompt = f"""
    Extract the city name from the following weather-related message. 
    If no specific city is mentioned, respond with "current location".
    If multiple cities are mentioned, pick the first one.
    Respond with only the city name, nothing else.

    Message: "{message_content}"

    City name:
    """

    result = lama3_2_llm.invoke([{"role": "user", "content": extraction_prompt}])
    city = result.content.strip()

    # Default to a major city if extraction fails or returns generic response
    if city.lower() in ["current location", "here", "my location", ""]:
        city = "New York"  # You can change this default

    return city


def weather_agent(state: State):
    last_message = state["messages"][-1]

    # Extract city from the message
    city = extract_city_from_message(last_message.content)

    # Get weather data
    weather_data = get_weather_data(city)
    #weather_data = get_weather_fake_data(city)

    if "error" in weather_data:
        response_content = f"Sorry, I couldn't get the weather information: {weather_data['error']}"
    else:
        # Format weather response
        response_content = f"""Weather in {weather_data['city']}, {weather_data['country']}:

🌡️ Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)
🌤️ Condition: {weather_data['description'].title()}
💧 Humidity: {weather_data['humidity']}%
🔽 Pressure: {weather_data['pressure']} hPa
💨 Wind Speed: {weather_data['wind_speed']} m/s"""

    return {"messages": [{"role": "assistant", "content": response_content}]}

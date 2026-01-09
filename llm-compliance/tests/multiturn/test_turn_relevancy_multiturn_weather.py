from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric
from dotenv import load_dotenv
load_dotenv()

# Example 1: Weather Planning Conversation
weather_planning_test = ConversationalTestCase(
    turns=[
        Turn(role="user", content="I'm planning a trip to New York next week"),
        Turn(role="assistant", content="I'd be happy to help you plan! What dates are you traveling?"),
        Turn(role="user", content="I'll be there from Monday to Friday"),
        Turn(role="assistant", content="Let me check the weather forecast for New York for those dates. I'll get the current conditions and forecast for you."),
        Turn(role="user", content="Great! What should I pack?"),
        Turn(role="assistant", content="Based on the forecast, I recommend packing layers - a light jacket, comfortable walking shoes, and an umbrella. The temperature will range from 15°C to 22°C with a chance of rain on Wednesday.")
    ]
)

# Example 2: Weather Comparison Conversation
weather_comparison_test = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What's the weather like in Paris today?"),
        Turn(role="assistant", content="Let me check the current weather in Paris for you. It's currently 18°C with partly cloudy skies."),
        Turn(role="user", content="How does that compare to London?"),
        Turn(role="assistant", content="London is slightly cooler at 16°C with light rain. Paris is a bit warmer and drier today."),
        Turn(role="user", content="Which city would be better for outdoor activities?"),
        Turn(role="assistant", content="Paris would be better for outdoor activities today since it's warmer and there's no rain, while London has light rain that might interfere with outdoor plans.")
    ]
)

# Example 3: Weather Troubleshooting Conversation
weather_troubleshooting_test = ConversationalTestCase(
    turns=[
        Turn(role="user", content="I'm supposed to have a picnic tomorrow but I'm worried about the weather"),
        Turn(role="assistant", content="I can help check the forecast! What location are you planning the picnic for?"),
        Turn(role="user", content="It's in Central Park, New York"),
        Turn(role="assistant", content="Let me check the weather forecast for Central Park, New York tomorrow. The forecast shows sunny skies with a high of 24°C and only a 10% chance of rain."),
        Turn(role="user", content="That sounds perfect! Should I bring anything special?"),
        Turn(role="assistant", content="The weather looks great! I'd suggest bringing sunscreen since it will be sunny, and maybe a light blanket as it might cool down in the shade. Otherwise, you should have perfect picnic weather!")
    ]
)

# Example 4: Multi-Location Weather Check Conversation
multi_location_test = ConversationalTestCase(
    turns=[
        Turn(role="user", content="I need to know the weather in Tokyo"),
        Turn(role="assistant", content="Let me check the current weather in Tokyo for you. It's currently 22°C with clear skies."),
        Turn(role="user", content="What about the weather in Seoul?"),
        Turn(role="assistant", content="Seoul is currently 19°C with partly cloudy conditions. It's slightly cooler than Tokyo."),
        Turn(role="user", content="Which city has better weather for sightseeing today?"),
        Turn(role="assistant", content="Both cities have good weather for sightseeing! Tokyo has clear skies which is perfect, while Seoul is partly cloudy but still pleasant. I'd say Tokyo has slightly better conditions with clearer skies, but both are great for outdoor activities.")
    ]
)

# Example 5: Weather Alert Conversation
weather_alert_test = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Is there any severe weather expected in Miami this weekend?"),
        Turn(role="assistant", content="Let me check the weather forecast for Miami this weekend. I'll look for any severe weather warnings or alerts."),
        Turn(role="user", content="I have outdoor plans on Saturday"),
        Turn(role="assistant", content="The forecast shows a 60% chance of thunderstorms on Saturday afternoon in Miami. I'd recommend having a backup indoor plan or rescheduling your outdoor activities to the morning when the chance of rain is lower."),
        Turn(role="user", content="What about Sunday?"),
        Turn(role="assistant", content="Sunday looks much better! It's forecasted to be partly sunny with only a 20% chance of rain, so Sunday would be a better day for your outdoor plans.")
    ]
)

# Initialize the metric
metric = TurnRelevancyMetric(threshold=0.5)

# Example of running metric as standalone on a single test case
print("=== Running Standalone Metric ===")
metric.measure(weather_planning_test)
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}\n")

# Run evaluation on all test cases
print("=== Running Full Evaluation ===")
evaluate(
    test_cases=[
        weather_planning_test,
        weather_comparison_test,
        weather_troubleshooting_test,
        multi_location_test,
        weather_alert_test
    ],
    metrics=[metric]
)


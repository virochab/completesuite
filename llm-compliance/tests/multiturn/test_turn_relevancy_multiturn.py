from deepeval import evaluate
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric
from dotenv import load_dotenv
load_dotenv()

# Example 1: Customer Support Conversation
customer_support_test = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Hi, I'm having trouble logging into my account"),
        Turn(role="assistant", content="I'd be happy to help you with that. Can you tell me what error message you're seeing?"),
        Turn(role="user", content="It says 'invalid password' but I'm sure I'm using the right one"),
        Turn(role="assistant", content="Let's try resetting your password. I'll send you a reset link to your registered email address."),
        Turn(role="user", content="Great, I received it and was able to log in. Thanks!"),
        Turn(role="assistant", content="Wonderful! I'm glad that worked. Is there anything else I can help you with today?")
    ]
)

# Example 2: Technical Support Conversation
tech_support_test = ConversationalTestCase(
    turns=[
        Turn(role="user", content="My printer won't connect to WiFi"),
        Turn(role="assistant", content="I can help troubleshoot that. First, is the printer powered on and showing any lights?"),
        Turn(role="user", content="Yes, it's on and the power light is green"),
        Turn(role="assistant", content="Good. Now let's check if WiFi is enabled on the printer. Can you navigate to the network settings on the printer's display?"),
        Turn(role="user", content="I found it. WiFi is turned off"),
        Turn(role="assistant", content="Perfect! Please turn WiFi on and select your network from the available networks list. You'll need to enter your WiFi password.")
    ]
)

# Example 3: Shopping Assistant Conversation
shopping_test = ConversationalTestCase(
    turns=[
        Turn(role="user", content="I'm looking for a laptop for college"),
        Turn(role="assistant", content="I'd be happy to help! What's your budget and what will you primarily use it for?"),
        Turn(role="user", content="Around $800, mainly for coding and watching videos"),
        Turn(role="assistant", content="For coding and media, I'd recommend a laptop with at least 16GB RAM and a good processor. Would you prefer Windows or Mac?"),
        Turn(role="user", content="Windows please"),
        Turn(role="assistant", content="Great choice! I recommend the Dell XPS 13 or HP Pavilion 15. Both have excellent specs for coding within your budget.")
    ]
)

# Example 4: Recipe Assistant Conversation
recipe_test = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What can I make with chicken, rice, and broccoli?"),
        Turn(role="assistant", content="You can make a delicious chicken and broccoli stir-fry over rice! Do you have any soy sauce or garlic?"),
        Turn(role="user", content="Yes, I have both"),
        Turn(role="assistant", content="Perfect! Sauté diced chicken with garlic, add broccoli florets, then toss with soy sauce. Serve over cooked rice."),
        Turn(role="user", content="How long should I cook the chicken?"),
        Turn(role="assistant", content="Cook the chicken for about 6-7 minutes until it's no longer pink inside and reaches 165°F internal temperature.")
    ]
)

# Initialize the metric
metric = TurnRelevancyMetric(threshold=0.5)

# Example of running metric as standalone on a single test case
print("=== Running Standalone Metric ===")
metric.measure(customer_support_test)
print(f"Score: {metric.score}")
print(f"Reason: {metric.reason}\n")

# Run evaluation on all test cases
print("=== Running Full Evaluation ===")
evaluate(
    test_cases=[
        customer_support_test,
     #   tech_support_test,
     #   shopping_test,
        recipe_test
    ],
    metrics=[metric]
)
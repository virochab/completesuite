from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import Pattern, PatternRecognizer
from detoxify import Detoxify

# Initialize engines
analyzer = AnalyzerEngine()

anonymizer = AnonymizerEngine()
def test_pii_anonymization():
    text = "Call me at 415-555-1212 tomorrow."
    results = analyzer.analyze(text=text, language="en")

    anonymized_text = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    ).text

    assert "415-555-1212" not in anonymized_text, "❌ Phone number not anonymized"
    print(f"✅ Anonymized text: {anonymized_text}")

test_pii_anonymization()


def sanitize_text(text: str) -> str:
    if not text or len(text.strip()) == 0:
        return text

    results = analyzer.analyze(text=text, language="en")
    anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized_result.text


def show_default_pii_entities():
    """Show all PII entities that Presidio detects by default"""
    print("🔍 Default PII entities detected by Presidio:")
    default_entities = [
        "PERSON",           # Names like "John Doe"
        "EMAIL_ADDRESS",    # Emails like "john@example.com"
        "PHONE_NUMBER",     # Phone numbers like "+1-555-123-4567"
        "CREDIT_CARD",      # Credit card numbers
        "SSN",              # Social Security Numbers
        "IBAN_CODE",        # International Bank Account Numbers
        "IP_ADDRESS",       # IP addresses like "192.168.1.1"
        "LOCATION",         # Addresses, cities, countries
        "DATE_TIME",        # Dates and times
        "NRP",              # Nationality, religious or political groups
        "MEDICAL_LICENSE",  # Medical license numbers
        "US_BANK_NUMBER",   # US bank account numbers
        "US_DRIVER_LICENSE", # US driver license numbers
        "US_PASSPORT",      # US passport numbers
        "UK_NHS",           # UK National Health Service numbers
        "URL",              # URLs and web addresses
        "US_ITIN",          # US Individual Taxpayer Identification Numbers
    ]
    
    for entity in default_entities:
        print(f"  ✓ {entity}")
    
    return default_entities


def validate_no_pii_in_text(text: str) -> tuple[bool, list]:
    """
    Validate that text doesn't contain PII data
    
    Args:
        text: Text to validate
        
    Returns:
        tuple: (is_clean, detected_entities)
    """
    results = analyzer.analyze(text=text, language="en")
    detected_entities = [r.entity_type for r in results]
    
    # Check if any PII entities were detected
    is_clean = len(results) == 0
    
    return is_clean, detected_entities


def test_pii_validation(anonymized_text: str):
    """Test PII validation on anonymized text"""
    print(f"Anonymized text: {anonymized_text}")
    
    # Validate anonymized text has no PII
    is_clean, detected_entities = validate_no_pii_in_text(anonymized_text)
    
    if is_clean:
        print("✅ Anonymized text contains no PII data")
    else:
        print(f"❌ Anonymized text still contains PII: {detected_entities}")
    
    # Assert that anonymized text is clean
    assert is_clean, f"Anonymized text contains PII: {detected_entities}"
    
    return is_clean, anonymized_text


def test_default_pii_detection():
    """Test default PII detection capabilities"""
    print("\n" + "="*60)
    print("TESTING DEFAULT PII DETECTION")
    print("="*60)
    
    # Show what entities are detected by default
    show_default_pii_entities()
    
    # Test with various PII types
    test_texts = [
        "My name is John Smith and I live in New York.",
        "Contact me at john.smith@company.com or call 555-123-4567",
        "My credit card is 4532-1234-5678-9012",
        "My SSN is 123-45-6789",
        "Visit our website at https://example.com",
        "I was born on January 15, 1990",
        "My driver's license is D123456789"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i}: {text} ---")
        results = analyzer.analyze(text=text, language="en")
        
        if results:
            entities = [r.entity_type for r in results]
            print(f"Detected PII: {entities}")
        else:
            print("No PII detected")


def test_pii_detection():
    text = "Hi, my name is John Doe. My SSN is 123-45-6789 and my email is john.doe@email.com"
    
    ssn_pattern = Pattern(name="ssn_pattern", regex=r"\d{3}-\d{2}-\d{4}", score=0.9)
    ssn_recognizer = PatternRecognizer(supported_entity="SSN_NUMBER", patterns=[ssn_pattern])

    analyzer.registry.add_recognizer(ssn_recognizer)
    # results = analyzer.analyze(text=text, language="en",entities=["EMAIL_ADDRESS","PERSON","SSN_NUMBER"])
    results = analyzer.analyze(text=text, language="en")
    print(results)
    entities = [r.entity_type for r in results]
    for entity in entities:
        print("entities: " + entity)
    
    assert "PERSON" in entities, "❌ Presidio failed to detect PERSON"
    assert "SSN_NUMBER" in entities, "❌ Presidio failed to detect SSN"
    assert "EMAIL_ADDRESS" in entities, "❌ Presidio failed to detect email"
    print("✅ PII detection test passed.")

    print(sanitize_text(text))
    # Original text with PII
    original_text = "Hi, my name is John Doe. My SSN is 123-45-6789 and my email is john.doe@email.com"
    # Anonymize the text
    anonymized_text = sanitize_text(original_text)
    print("anonymized_text: " + anonymized_text)
    # Test PII validation on anonymized text
    test_pii_validation(anonymized_text)
    

test_pii_detection()


# Load the default model (English)
model = Detoxify('original')

# Example text to check
texts = [
    "I hate you so much!",
    "Have a nice day, friend.",
    "You're such an idiot.",
    "You're such an idiot. Asian",
    "how many times have to tell you this",
    "you are from India, which is in Asia so you are an asian by default"
]

# Run toxicity detection
for text in texts:
    scores = model.predict(text)
    print(f"\nText: {text}")
    for label, score in scores.items():
        print(f"  {label}: {score:.4f}")

import requests
import json
import sys

# --- CONFIG ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def load_json_config(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except: return {"MAX_CHARS": 90, "BANNED_WORDS": []}

def read_text_file(filepath):
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            return f.read().strip()
    except: return ""

def print_step_header(step_name):
    print("\n" + "="*60)
    print(f"🔹 {step_name}")
    print("="*60)

# --- STEP 1: PYTHON LOGIC (Hard Rules) ---
def run_step_1(text, config):
    issues = []
    if len(text) > config.get('MAX_CHARS', 90):
        issues.append(f"Hard Rule: Ad length ({len(text)}) exceeds {config.get('MAX_CHARS')} limit.")
    for word in config.get('BANNED_WORDS', []):
        if word.lower() in text.lower():
            issues.append(f"Hard Rule: Banned word detected: '{word}'")
    
    result = {
        "status": "FAIL" if issues else "PASS",
        "hard_violations": issues
    }
    print_step_header("STEP 1: HARD RULES (LOGIC)")
    print(json.dumps(result, indent=4))
    return result["status"]

# --- STEP 2: COMPLIANCE AI (Strict Guidelines) ---
def run_step_2(ad_text, biz_info, guidelines):
    prompt = f"""
    Analyze this ad against Business Info and Compliance Guidelines.
    Return ONLY a JSON object:
    {{
      "status": "PASS" or "FAIL",
      "issues": ["list every violation found"],
      "suggestions": ["how to fix the ad to make it compliant"]
    }}
    BUSINESS INFO: {biz_info}
    GUIDELINES: {guidelines}
    AD TEXT: {ad_text}
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "format": "json", "stream": False}
    
    print_step_header("STEP 2: COMPLIANCE CHECK (AI)")
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        result = json.loads(response.json()['response'])
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Error: {e}")

# --- STEP 3: RECOMMENDATION AI (Creative Suggestions) ---
def run_step_3(ad_text, biz_info, recommendations):
    # Added biz_info to the prompt here
    prompt = f"""
    You are a Senior Marketing Copywriter. 
    Analyze the Ad Copy using the Business Info and the Best Practice Recommendations.
    
    Return ONLY a JSON object:
    {{
      "improvement_score": "0-10",
      "missing_elements": ["list what the ad is missing from the recommendations"],
      "creative_suggestions": ["specific ways to improve the copy based on this specific business info"]
    }}
    
    BUSINESS INFO: {biz_info}
    RECOMMENDATIONS: {recommendations}
    AD TEXT: {ad_text}
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "format": "json", "stream": False}
    
    print_step_header("STEP 3: MARKETING RECOMMENDATIONS (AI)")
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        result = json.loads(response.json()['response'])
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"Error: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Load all 4 data sources and the config
    config = load_json_config("rules_config.json")
    ad_copy = read_text_file("ad_copy.txt")
    biz_info = read_text_file("business_info.txt")
    rules = read_text_file("guidelines.txt")
    recs = read_text_file("recommendation.txt")

    # Start Sequence
    # We check hard rules first
    status = run_step_1(ad_copy, config)
    
    # If the ad passes basic logic, we ask the AI for deeper analysis
    if status == "PASS":
        # Step 2: Compliance (Strict)
        run_step_2(ad_copy, biz_info, rules)
        
        # Step 3: Marketing (Creative - now includes Business Info)
        run_step_3(ad_copy, biz_info, recs)
    else:
        print("\n❌ Stopping: Fix Hard Rule violations (Step 1) before running AI analysis.")
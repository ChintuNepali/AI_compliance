import requests
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# --- CONFIG ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3-flash-preview"

# Configure Gemini
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

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

def load_image(filepath):
    """Load image file if it exists"""
    try:
        if Path(filepath).exists():
            return Image.open(filepath)
    except Exception as e:
        print(f"Warning: Could not load image {filepath}: {e}")
    return None

def parse_json_response(response_text):
    """Extract JSON from Gemini response, removing markdown code blocks"""
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

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

# --- STEP 1.5: IMAGE ANALYSIS WITH GEMINI (if image provided) ---
def run_image_analysis(image, ad_text, biz_info, guidelines):
    """Analyze image content using Gemini's multimodal capabilities"""
    if not image:
        return "PASS"
    
    print_step_header("STEP 1.5: IMAGE ANALYSIS (GEMINI)")
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""
Analyze this advertising image along with its text copy for Google Ads compliance.

AD TEXT: {ad_text}
BUSINESS INFO: {biz_info}
COMPLIANCE GUIDELINES (key points): {guidelines[:2000]}

Return ONLY a JSON object with:
{{
  "image_description": "brief description of what's in the image",
  "compliance_status": "PASS" or "FAIL",
  "image_issues": ["list any visual compliance violations - inappropriate content, misleading visuals, etc."],
  "text_image_alignment": ["does the image match the text claims? any discrepancies?"],
  "visual_quality": ["technical quality issues - resolution, cropping, professional appearance"],
  "recommendations": ["suggestions to improve the image for better compliance and performance"]
}}
"""
        
        response = model.generate_content([prompt, image])
        result = parse_json_response(response.text)
        print(json.dumps(result, indent=4))
        
        return result.get("compliance_status", "PASS")
        
    except Exception as e:
        print(f"Error in image analysis: {e}")
        return "PASS"

# --- STEP 2: COMPLIANCE AI (Strict Guidelines) ---
def run_step_2(ad_text, biz_info, guidelines, image=None):
    """Compliance check using Gemini (multimodal if image provided)"""
    print_step_header("STEP 2: COMPLIANCE CHECK (GEMINI)")
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
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
        
        # Use multimodal if image provided, otherwise text-only
        content = [prompt, image] if image else [prompt]
        response = model.generate_content(content)
        result = parse_json_response(response.text)
        print(json.dumps(result, indent=4))
        
    except Exception as e:
        print(f"Error in compliance check: {e}")

# --- STEP 3: RECOMMENDATION AI (Creative Suggestions) ---
def run_step_3(ad_text, biz_info, recommendations, image=None):
    """Marketing recommendations using Gemini (multimodal if image provided)"""
    print_step_header("STEP 3: MARKETING RECOMMENDATIONS (GEMINI)")
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        prompt = f"""
You are a Senior Marketing Copywriter.
Analyze the Ad Copy using the Business Info and the Best Practice Recommendations.

Return ONLY a JSON object:
{{
  "improvement_score": "0-10",
  "missing_elements": ["list what the ad is missing from the recommendations"],
  "creative_suggestions": ["specific ways to improve the copy based on this specific business info"]{', "visual_text_synergy": ["how well do the image and text work together?"]' if image else ''}
}}

BUSINESS INFO: {biz_info}
RECOMMENDATIONS: {recommendations}
AD TEXT: {ad_text}
"""
        
        # Use multimodal if image provided, otherwise text-only
        content = [prompt, image] if image else [prompt]
        response = model.generate_content(content)
        result = parse_json_response(response.text)
        print(json.dumps(result, indent=4))
        
    except Exception as e:
        print(f"Error in marketing recommendations: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Load all data sources
    config = load_json_config("rules_config.json")
    ad_copy = read_text_file("ad_copy.txt")
    biz_info = read_text_file("business_info.txt")
    rules = read_text_file("guidelines.txt")
    recs = read_text_file("recommendation.txt")
    
    # Try to load ad image (common formats)
    ad_image = None
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
        ad_image = load_image(f"ad_image{ext}")
        if ad_image:
            print(f"✓ Loaded ad image: ad_image{ext}")
            break
    
    if not ad_image:
        print("ℹ No ad image found (looking for ad_image.jpg/png/gif/webp)")
    
    # Start Sequence
    status = run_step_1(ad_copy, config)
    
    if status == "PASS":
        # Step 1.5: Analyze image separately if provided
        if ad_image:
            image_status = run_image_analysis(ad_image, ad_copy, biz_info, rules)
            if image_status == "FAIL":
                print("\n⚠️ Image compliance issues detected. Review image violations above.")
        
        # Step 2: Compliance (with or without image)
        run_step_2(ad_copy, biz_info, rules, ad_image)
        
        # Step 3: Marketing recommendations (with or without image)
        run_step_3(ad_copy, biz_info, recs, ad_image)
    else:
        print("\n❌ Stopping: Fix Hard Rule violations (Step 1) before running AI analysis.")
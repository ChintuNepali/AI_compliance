"""
FastAPI backend for AdChecker - Ad Compliance Analysis API
"""
import json
import os
import io
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# Initialize FastAPI
app = FastAPI(
    title="AdChecker API",
    description="AI-powered ad compliance checker for Google Ads",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from web folder
app.mount("/static", StaticFiles(directory="web"), name="static")


def load_json_config(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return {"MAX_CHARS": 90, "BANNED_WORDS": []}


def read_text_file(filepath):
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            return f.read().strip()
    except:
        return ""


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


# --- STEP 1: Hard Rules Check ---
def check_hard_rules(text: str, config: dict) -> dict:
    issues = []
    if len(text) > config.get('MAX_CHARS', 90):
        issues.append(f"Ad length ({len(text)}) exceeds {config.get('MAX_CHARS')} character limit.")
    for word in config.get('BANNED_WORDS', []):
        if word.lower() in text.lower():
            issues.append(f"Banned word detected: '{word}'")
    
    return {
        "step": "Hard Rules Check",
        "status": "FAIL" if issues else "PASS",
        "violations": issues
    }


# --- STEP 1.5: Image Analysis ---
async def analyze_image(image: Image.Image, ad_text: str, biz_info: str, guidelines: str) -> dict:
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
        result["step"] = "Image Analysis"
        return result
        
    except Exception as e:
        return {
            "step": "Image Analysis",
            "error": str(e),
            "compliance_status": "ERROR"
        }


# --- STEP 2: Compliance Check ---
async def check_compliance(ad_text: str, biz_info: str, guidelines: str, image: Optional[Image.Image] = None) -> dict:
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
        
        content = [prompt, image] if image else [prompt]
        response = model.generate_content(content)
        result = parse_json_response(response.text)
        result["step"] = "Compliance Check"
        return result
        
    except Exception as e:
        return {
            "step": "Compliance Check",
            "error": str(e),
            "status": "ERROR"
        }


# --- STEP 3: Marketing Recommendations ---
async def get_recommendations(ad_text: str, biz_info: str, recommendations: str, image: Optional[Image.Image] = None) -> dict:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        visual_field = ', "visual_text_synergy": ["how well do the image and text work together?"]' if image else ''
        
        prompt = f"""
You are a Senior Marketing Copywriter.
Analyze the Ad Copy using the Business Info and the Best Practice Recommendations.

Return ONLY a JSON object:
{{
  "improvement_score": "0-10",
  "missing_elements": ["list what the ad is missing from the recommendations"],
  "creative_suggestions": ["specific ways to improve the copy based on this specific business info"]{visual_field}
}}

BUSINESS INFO: {biz_info}
RECOMMENDATIONS: {recommendations}
AD TEXT: {ad_text}
"""
        
        content = [prompt, image] if image else [prompt]
        response = model.generate_content(content)
        result = parse_json_response(response.text)
        result["step"] = "Marketing Recommendations"
        return result
        
    except Exception as e:
        return {
            "step": "Marketing Recommendations",
            "error": str(e),
            "improvement_score": "N/A"
        }


# --- API ENDPOINTS ---

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    return FileResponse("web/index.html")


@app.get("/api/config")
async def get_config():
    """Get current rules configuration"""
    config = load_json_config("rules_config.json")
    return config


@app.get("/api/defaults")
async def get_defaults():
    """Get default text file contents for the form"""
    return {
        "business_info": read_text_file("business_info.txt"),
        "guidelines": read_text_file("guidelines.txt"),
        "recommendations": read_text_file("recommendation.txt")
    }


@app.post("/api/analyze")
async def analyze_ad(
    ad_text: str = Form(...),
    business_info: str = Form(...),
    guidelines: str = Form(...),
    recommendations: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """
    Main endpoint: Analyze ad copy (and optional image) for compliance
    """
    results = {
        "ad_text": ad_text,
        "has_image": image is not None,
        "steps": []
    }
    
    # Load config for hard rules
    config = load_json_config("rules_config.json")
    
    # Load image if provided
    pil_image = None
    if image:
        try:
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes))
            results["image_filename"] = image.filename
        except Exception as e:
            results["image_error"] = str(e)
    
    # STEP 1: Hard Rules
    step1 = check_hard_rules(ad_text, config)
    results["steps"].append(step1)
    
    # If hard rules fail, stop here
    if step1["status"] == "FAIL":
        results["overall_status"] = "FAIL"
        results["message"] = "Fix hard rule violations before proceeding with AI analysis."
        return results
    
    # STEP 1.5: Image Analysis (if image provided)
    if pil_image:
        step1_5 = await analyze_image(pil_image, ad_text, business_info, guidelines)
        results["steps"].append(step1_5)
    
    # STEP 2: Compliance Check
    step2 = await check_compliance(ad_text, business_info, guidelines, pil_image)
    results["steps"].append(step2)
    
    # STEP 3: Marketing Recommendations
    step3 = await get_recommendations(ad_text, business_info, recommendations, pil_image)
    results["steps"].append(step3)
    
    # Determine overall status
    statuses = [s.get("status") or s.get("compliance_status") for s in results["steps"]]
    results["overall_status"] = "FAIL" if "FAIL" in statuses else "PASS"
    
    return results


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# main_updated.py — Part 1/5
import os
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, EmailStr
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from datetime import datetime, timedelta
import google.generativeai as genai
from bson import ObjectId
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import asyncio
from upstash_redis.asyncio import Redis
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
import random
import requests  # external RAG API calls

# ============================
# 🔧 Environment Configuration (Railway)
# ============================

# Load environment variables
REDIS_URL = os.getenv("REDIS_URL")
REDIS_TOKEN = os.getenv("REDIS_TOKEN")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "fastapi_demo_db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")

# Validate required envs
required_env = {
    "REDIS_URL": REDIS_URL,
    "REDIS_TOKEN": REDIS_TOKEN,
    "MONGO_URI": MONGO_URI,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "BREVO_API_KEY": BREVO_API_KEY
}

missing_env = [k for k, v in required_env.items() if v is None]
if missing_env:
    raise RuntimeError(f"❌ Missing environment variables: {', '.join(missing_env)}")

# ============================
# 🔧 Redis Setup
# ============================
try:
    redis = Redis(url=REDIS_URL, token=REDIS_TOKEN)
    REDIS_AVAILABLE = True
    print("✅ Upstash Redis Connected")
except Exception as e:
    print(f"❌ Upstash Redis failed: {e}")
    REDIS_AVAILABLE = False
    redis = None

# ============================
# 🔧 Embedding Model
# ============================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================
# 🔧 MongoDB Setup
# ============================
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
users_collection = db["users"]
sessions_collection = db["chat_sessions"]
global_qa_collection = db["global_qa"]
otp_collection = db["otp_verification"]
feedback_collection = db["user_feedback"]

# Indexes
users_collection.create_index("email", unique=True)
global_qa_collection.create_index("normalized_question", unique=True)
global_qa_collection.create_index("created_at")
otp_collection.create_index("email", unique=True)
otp_collection.create_index("created_at", expireAfterSeconds=600)
feedback_collection.create_index("email", unique=True)

# ============================
# 🔧 Gemini Setup
# ============================
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ============================
# 🔧 Brevo Email Setup
# ============================
configuration = sib_api_v3_sdk.Configuration()
configuration.api_key["api-key"] = BREVO_API_KEY
api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
    sib_api_v3_sdk.ApiClient(configuration)
)

# ============================
# ⚙️ FastAPI App Setup
# ============================
app = FastAPI(
    title="Medical Chat API",
    description="FastAPI + Gemini 2.5 Flash + MongoDB + Redis + OTP + Feedback"
)

# ============================
# 📘 Pydantic Models
# ============================
class User(BaseModel):
    email: EmailStr
    password: str

class OTPRequest(BaseModel):
    email: EmailStr

class OTPVerify(BaseModel):
    email: EmailStr
    otp: int
    password: str

class AskRequest(BaseModel):
    email: EmailStr
    session_id: str | None = None
    question: str

class FeedbackRequest(BaseModel):
    email: EmailStr
    feedback: List[str]

class DeleteAccountRequest(BaseModel):
    email: EmailStr


# ============================
# 🔧 Utility Functions
# ============================
def normalize_text(text: str) -> str:
    """Normalize text for string matching"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

def extract_name_from_email(email: str) -> str:
    """Extract name from email (part before @)"""
    return email.split('@')[0].capitalize()

async def send_welcome_email_internal(email: str, name: str):
    """Internal function to send welcome email"""
    html_content = f"""
    <div style="font-family: Arial, sans-serif; color: #222;">
        <p>Hi {name},</p>
        <p>
            Welcome to <b>MedNexus</b> — your AI-powered medical support assistant.
        </p>
        <p>
            MedNexus uses advanced RAG, CAG (Catch Augmented Generation), and
            Agentic Context Enhancement to deliver accurate medical information
            from trusted datasets such as PubMedQA and MedQA.
        </p>
        <p>
            You can ask any medical-related question, and we will respond with
            clear and reliable answers.
        </p>
        <p>
            We're glad to have you with us.<br>
            — Team MedNexus
        </p>
    </div>
    """
    email_data = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": email}],
        sender={"name": "MedNexus", "email": "mrsadiq471@gmail.com"},
        subject=f"Welcome to MedNexus, {name}",
        html_content=html_content,
        text_content=f"Welcome {name}! Thanks for joining MedNexus."
    )
    try:
        api_instance.send_transac_email(email_data)
        print(f"Welcome email sent to {email}")
    except ApiException as e:
        print(f"Failed to send welcome email: {str(e)}")
# main_updated.py — Part 2/5 (continuation)
async def get_user_feedback(email: str) -> List[str]:
    """Get user feedback from MongoDB"""
    feedback_doc = feedback_collection.find_one({"email": email})
    if feedback_doc and "feedback" in feedback_doc:
        return feedback_doc["feedback"]
    return []

async def user_has_feedback(email: str) -> bool:
    """Check if user has any feedback"""
    feedback_doc = feedback_collection.find_one({"email": email})
    return feedback_doc is not None and "feedback" in feedback_doc and len(feedback_doc["feedback"]) > 0

async def store_feedback_in_redis(email: str, feedback: List[str]):
    """Store feedback in Redis with error handling"""
    if not REDIS_AVAILABLE or redis is None:
        return
    try:
        redis_key = f"feedback:{email}"
        await redis.setex(
            redis_key,
            3600,  # 1 hour TTL
            json.dumps({
                "feedback": feedback,
                "updated_at": datetime.utcnow().isoformat()
            })
        )
    except Exception as e:
        print(f"Redis storage failed: {e}. Continuing without Redis...")

async def get_feedback_from_redis(email: str) -> Optional[Dict[str, Any]]:
    """Get feedback from Redis with error handling"""
    if not REDIS_AVAILABLE or redis is None:
        return None
    try:
        redis_key = f"feedback:{email}"
        cached_feedback = await redis.get(redis_key)
        if cached_feedback:
            return json.loads(cached_feedback)
    except Exception as e:
        print(f"Redis fetch failed: {e}")
    return None

async def delete_feedback_from_redis(email: str):
    """Delete feedback from Redis with error handling"""
    if not REDIS_AVAILABLE or redis is None:
        return
    try:
        redis_key = f"feedback:{email}"
        await redis.delete(redis_key)
    except Exception as e:
        print(f"Redis delete failed: {e}")

async def find_similar_question_in_redis(user_question: str, threshold: float = 0.85) -> Optional[Dict[str, Any]]:
    """Search for similar questions using MongoDB and return context+source when available"""
    normalized_question = normalize_text(user_question)

    # Exact match
    exact_match = global_qa_collection.find_one({"normalized_question": normalized_question})
    if exact_match:
        return {
            'question': exact_match['question'],
            'answer': exact_match['answer'],
            'context': exact_match.get('context'),
            'source': exact_match.get('source'),
            'similarity_type': 'exact_match',
            'similarity_score': 1.0
        }

    # Fuzzy string matching
    all_qa = list(global_qa_collection.find().limit(1000))

    for qa in all_qa:
        stored_normalized = qa.get('normalized_question', '')
        stored_words = set(stored_normalized.split())
        user_words = set(normalized_question.split())
        common_words = stored_words.intersection(user_words)

        if len(common_words) >= max(2, int(min(len(stored_words), len(user_words)) * 0.8)):
            return {
                'question': qa['question'],
                'answer': qa['answer'],
                'context': qa.get('context'),
                'source': qa.get('source'),
                'similarity_type': 'fuzzy_match',
                'similarity_score': len(common_words) / max(len(stored_words), len(user_words))
            }

    # Semantic search
    user_embedding = embedding_model.encode([user_question])[0].tolist()
    best_match = None
    best_similarity = 0.0

    for qa in all_qa:
        if 'embedding' in qa:
            stored_embedding = qa['embedding']
            similarity = cosine_similarity(user_embedding, stored_embedding)
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = {
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'context': qa.get('context'),
                    'source': qa.get('source'),
                    'similarity_type': 'semantic_match',
                    'similarity_score': similarity
                }

    return best_match

async def store_qa_in_redis_and_mongo(question: str, answer: str, context: Optional[str], source: Optional[str], embedding: List[float]):
    """Store Q&A pair (including context + source) in MongoDB"""
    normalized_question = normalize_text(question)

    global_qa_doc = {
        'question': question,
        'normalized_question': normalized_question,
        'answer': answer,
        'context': context,
        'source': source,
        'embedding': embedding,
        'created_at': datetime.utcnow()
    }

    try:
        global_qa_collection.insert_one(global_qa_doc)
    except DuplicateKeyError:
        global_qa_collection.update_one(
            {"normalized_question": normalized_question},
            {"$set": {
                "answer": answer,
                "context": context,
                "source": source,
                "embedding": embedding,
                "updated_at": datetime.utcnow()
            }}
        )

def store_chat_in_db(session_id: str, chat_entry: dict):
    """Background task to store chat in MongoDB"""
    try:
        sessions_collection.update_one(
            {"_id": ObjectId(session_id)},
            {"$push": {"history": chat_entry}}
        )
    except Exception as e:
        print(f"Error storing chat in DB: {e}")
# main_updated.py — Part 3/5 (continuation)
# ============================
# 🩺 Health Check
# ============================
@app.get("/health")
async def health_check():
    redis_status = "unavailable"
    if REDIS_AVAILABLE and redis:
        try:
            await redis.ping()
            redis_status = "healthy"
        except Exception as e:
            redis_status = f"unhealthy: {str(e)}"

    try:
        await asyncio.get_event_loop().run_in_executor(None, users_collection.find_one)
        mongodb_status = "healthy"
    except Exception as e:
        mongodb_status = f"unhealthy: {str(e)}"

    return {
        "status": "ok",
        "message": "Server is healthy",
        "redis": redis_status,
        "mongodb": mongodb_status,
        "redis_available": REDIS_AVAILABLE
    }

# ============================
# 🔐 OTP-Based Registration Flow
# ============================
@app.post("/send-otp")
async def send_otp(request: OTPRequest):
    """Step 1: Send OTP to user's email"""
    email = request.email

    # Check if user already exists
    existing_user = users_collection.find_one({"email": email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    # Generate OTP
    otp = random.randint(100000, 999999)

    # Store OTP in MongoDB (will auto-delete after 10 minutes)
    otp_collection.update_one(
        {"email": email},
        {"$set": {
            "otp": otp,
            "created_at": datetime.utcnow()
        }},
        upsert=True
    )

    # Send OTP email
    html_content = f"""
    <div style="font-family: Arial, sans-serif; color: #222;">
        <p>Your MedNexus verification code:</p>
        <h2 style="letter-spacing: 4px;">{otp}</h2>
        <p>This code is valid for 10 minutes.</p>
        <p>— Team MedNexus</p>
    </div>
    """
    email_data = sib_api_v3_sdk.SendSmtpEmail(
        to=[{"email": email}],
        sender={"name": "MedNexus", "email": "mrsadiq471@gmail.com"},
        subject="Your MedNexus verification code",
        html_content=html_content,
        text_content=f"Your MedNexus OTP is {otp}. It is valid for 10 minutes."
    )

    try:
        response = api_instance.send_transac_email(email_data)
        return {
            "status": "success",
            "message": "OTP sent to your email",
            "brevo_id": response.message_id
        }
    except ApiException as e:
        raise HTTPException(status_code=500, detail=f"Failed to send OTP: {str(e)}")

@app.post("/verify-otp-and-register")
async def verify_otp_and_register(request: OTPVerify, background_tasks: BackgroundTasks):
    """Step 2: Verify OTP and create account"""
    email = request.email
    otp = request.otp
    password = request.password

    # Retrieve stored OTP from MongoDB
    otp_record = otp_collection.find_one({"email": email})

    if not otp_record:
        raise HTTPException(status_code=400, detail="OTP not found or expired")

    # Check if OTP matches
    if otp_record["otp"] != otp:
        return {
            "status": "error",
            "verified": False,
            "message": "Invalid OTP"
        }

    # Check if OTP is still valid (within 10 minutes)
    otp_created_at = otp_record["created_at"]
    if datetime.utcnow() - otp_created_at > timedelta(minutes=10):
        otp_collection.delete_one({"email": email})
        raise HTTPException(status_code=400, detail="OTP expired")

    # OTP is valid - Create user account
    try:
        users_collection.insert_one({
            "email": email,
            "password": password,
            "created_at": datetime.utcnow(),
            "has_feedback": False
        })

        # Delete OTP record after successful registration
        otp_collection.delete_one({"email": email})

        # Extract name from email and send welcome email in background
        name = extract_name_from_email(email)
        background_tasks.add_task(send_welcome_email_internal, email, name)

        return {
            "status": "success",
            "verified": True,
            "message": "Account created successfully",
            "email": email
        }

    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="User already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# ============================
# 🔐 Login
# ============================
@app.post("/login")
def login(user: User):
    found_user = users_collection.find_one({"email": user.email})
    if not found_user:
        raise HTTPException(status_code=404, detail="User not found")
    if found_user["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"message": "Login successful", "email": user.email}

# ============================
# 📝 Feedback Management APIs
# ============================
@app.post("/submit-feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit or update user feedback (replaces existing feedback)"""
    email = request.email

    # Check if user exists
    user = users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Store/update feedback in MongoDB
    feedback_doc = {
        "email": email,
        "feedback": request.feedback,
        "updated_at": datetime.utcnow(),
        "has_feedback": len(request.feedback) > 0
    }

    try:
        # Update or insert feedback
        feedback_collection.update_one(
            {"email": email},
            {"$set": feedback_doc},
            upsert=True
        )

        # Update user's has_feedback status
        users_collection.update_one(
            {"email": email},
            {"$set": {"has_feedback": len(request.feedback) > 0}}
        )

        # Store in Redis (if available)
        await store_feedback_in_redis(email, request.feedback)

        return {
            "status": "success",
            "message": "Feedback submitted successfully",
            "email": email,
            "feedback_count": len(request.feedback),
            "has_feedback": len(request.feedback) > 0,
            "redis_available": REDIS_AVAILABLE
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")
# main_updated.py — Part 4/5 (continuation)
@app.get("/get-feedback")
async def get_feedback(email: str = Query(..., description="User email")):
    """Get all feedback for a user"""
    # Check if user exists
    user = users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Try to get from Redis first
    cached_feedback = await get_feedback_from_redis(email)
    if cached_feedback:
        return {
            "email": email,
            "feedback": cached_feedback["feedback"],
            "has_feedback": len(cached_feedback["feedback"]) > 0,
            "updated_at": cached_feedback["updated_at"],
            "source": "redis",
            "redis_available": REDIS_AVAILABLE
        }

    # Fallback to MongoDB
    feedback_doc = feedback_collection.find_one({"email": email})
    if feedback_doc and "feedback" in feedback_doc:
        return {
            "email": email,
            "feedback": feedback_doc["feedback"],
            "has_feedback": feedback_doc.get("has_feedback", len(feedback_doc["feedback"]) > 0),
            "updated_at": feedback_doc.get("updated_at", datetime.utcnow().isoformat()),
            "source": "mongodb",
            "redis_available": REDIS_AVAILABLE
        }
    else:
        return {
            "email": email,
            "feedback": [],
            "has_feedback": False,
            "updated_at": datetime.utcnow().isoformat(),
            "source": "none",
            "redis_available": REDIS_AVAILABLE
        }

@app.get("/user-has-feedback")
async def check_user_has_feedback(email: str = Query(..., description="User email")):
    """Check if user has any feedback"""
    user = users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    has_feedback = await user_has_feedback(email)
    return {
        "email": email,
        "has_feedback": has_feedback,
        "redis_available": REDIS_AVAILABLE
    }

# ============================
# 🗑️ Delete Account API
# ============================
@app.delete("/delete-account")
async def delete_account(request: DeleteAccountRequest):
    """Delete user account and all associated data"""
    email = request.email

    try:
        # Delete user from users collection
        user_result = users_collection.delete_one({"email": email})
        if user_result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        # Delete all user sessions
        sessions_result = sessions_collection.delete_many({"email": email})

        # Delete user feedback
        feedback_result = feedback_collection.delete_one({"email": email})

        # Delete from Redis cache
        await delete_feedback_from_redis(email)

        return {
            "status": "success",
            "message": "Account and all associated data deleted successfully",
            "email": email,
            "sessions_deleted": sessions_result.deleted_count,
            "feedback_deleted": feedback_result.deleted_count if feedback_result else 0,
            "redis_available": REDIS_AVAILABLE
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete account: {str(e)}")
# main_updated.py — Part 5/5 (final)
# ============================
# 💬 Ask Medical Question (Updated with Feedback Integration)
# ============================
@app.post("/ask-medical")
async def ask_medical(req: AskRequest, background_tasks: BackgroundTasks):
    """Ask medical question with caching, session management, feedback integration"""
    similar_qa = await find_similar_question_in_redis(req.question)

    # Handle session creation
    if not req.session_id:
        session_doc = {
            "email": req.email,
            "created_at": datetime.utcnow(),
            "history": []
        }
        session_id = str(sessions_collection.insert_one(session_doc).inserted_id)
    else:
        session_id = req.session_id
        session_doc = sessions_collection.find_one({"_id": ObjectId(session_id), "email": req.email})
        if not session_doc:
            raise HTTPException(status_code=404, detail="Invalid session or email mismatch")

    # Return cached response if found (include context & source)
    if similar_qa:
        chat_entry = {
            "question": req.question,
            "answer": similar_qa["answer"],
            "context": similar_qa.get("context"),
            "source": similar_qa.get("source"),
            "is_follow_up": False,
            "cache_hit": True,
            "similarity_type": similar_qa["similarity_type"],
            "similarity_score": similar_qa["similarity_score"],
            "timestamp": datetime.utcnow()
        }

        background_tasks.add_task(store_chat_in_db, session_id, chat_entry)

        return {
            "session_id": session_id,
            "answer": similar_qa["answer"],
            "context": similar_qa.get("context"),
            "source": similar_qa.get("source"),
            "cache_hit": True,
            "similarity_type": similar_qa["similarity_type"],
            "similarity_score": similar_qa["similarity_score"]
        }

    # Get previous questions for context
    session_doc = sessions_collection.find_one({"_id": ObjectId(session_id), "email": req.email})
    history = session_doc.get("history", [])
    previous_questions = []
    for entry in history[-2:]:
        if "original_question" in entry:
            previous_questions.append(entry["original_question"])
        else:
            previous_questions.append(entry["question"])

    # Get user feedback for personalized responses
    user_feedback = await get_user_feedback(req.email)
    has_feedback = await user_has_feedback(req.email)

    # Update user's has_feedback status
    if has_feedback:
        users_collection.update_one(
            {"email": req.email},
            {"$set": {"has_feedback": True}}
        )

    # Prepare prompt with feedback integration
    feedback_context = ""
    if user_feedback:
        feedback_context = f"""
User Preferences & Feedback:
{chr(10).join([f"• {fb}" for fb in user_feedback])}

Important: Please incorporate these preferences into your response style and content approach.
"""

    if previous_questions:
        prompt = f"""
You are MedBot — a professional AI medical assistant.
{feedback_context}
Context:
Recent questions: {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(reversed(previous_questions))])}
Current: "{req.question}"
Tasks:
1. Detect if this is a follow-up to the above context.
2. If follow-up → rewrite as a complete standalone medical question.
3. Answer the question clearly and medically accurate, considering user preferences if provided.
4. Output must be **only JSON**, no extra text.
JSON format:
{{
  "is_follow_up": true/false,
  "reformatted_question": "if follow-up, else empty",
  "medical_answer": "clear, concise paragraph incorporating user preferences"
}}
Rules:
- Only answer medical/health questions.
- For non-medical ones: "Sorry, I don't have any information about this."
- Avoid emojis, markdown, or symbols.
- Consider user feedback/preferences in your response style.
"""
    else:
        prompt = f"""
You are MedBot — a professional AI medical assistant.
{feedback_context}
Question: "{req.question}"
Tasks:
1. Provide a concise, medically accurate paragraph answer considering user preferences.
2. Respond only in JSON (no extra text).
JSON format:
{{
  "is_follow_up": false,
  "reformatted_question": "",
  "medical_answer": "clear, concise paragraph incorporating user preferences"
}}
Rules:
- Answer only medical/health questions.
- If non-medical: "Sorry, I don't have any information about this."
- No markdown, no emojis, no symbols.
- Consider user feedback/preferences in your response style.
"""

    # Query Gemini
    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        try:
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            response_data = json.loads(response_text)

            is_follow_up = response_data.get("is_follow_up", False)
            reformatted_question = response_data.get("reformatted_question", "").strip()
            medical_answer = response_data.get("medical_answer", "").strip()

            if req.question in previous_questions:
                is_follow_up = False
                reformatted_question = ""

        except json.JSONDecodeError:
            is_follow_up = False
            reformatted_question = ""
            medical_answer = "Sorry, I encountered an error. Please try again."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

    # Detect non-medical fallback via phrase matching (no 'source' variable used for Gemini)
    non_medical_phrases = [
        "sorry, i don't have any information about this",
        "i cannot answer that",
        "this is not a medical question",
        "i don't have any information about this",
        "i don't have information about this",
        "sorry, i cannot answer that",
        "this is outside my medical knowledge",
        "i can only answer medical questions"
    ]
    is_medical = not any(phrase in medical_answer.lower() for phrase in non_medical_phrases)

    final_question = reformatted_question if (is_follow_up and reformatted_question) else req.question
    question_embedding = embedding_model.encode([final_question])[0].tolist()

    # ================================
    # 🔥 CALL EXTERNAL RAG API (if medical)
    # ================================
    external_answer = None
    external_source = None

    if is_medical:
        try:
            api_url = "https://mednexusrag-production.up.railway.app/ask"
            ext_res = requests.get(api_url, params={"question": final_question}, timeout=15)
            if ext_res.status_code == 200:
                ext_json = ext_res.json()
                external_answer = ext_json.get("answer")
                external_source = ext_json.get("source")
            else:
                external_answer = None
                external_source = None
        except Exception:
            external_answer = None
            external_source = None

    # Store in cache if medical question (FIXED: store gemini answer as answer, external as context)
    if is_medical:
        to_store_answer = medical_answer
        context_to_store = external_answer
        background_tasks.add_task(
            store_qa_in_redis_and_mongo,
            final_question,
            to_store_answer,         # answer to store (Gemini)
            context_to_store,        # context (external)
            external_source,         # source
            question_embedding
        )

    # Prepare chat entry (include context & source)
    if is_follow_up and reformatted_question:
        chat_entry = {
            "original_question": req.question,
            "reformatted_question": reformatted_question,
            "answer": medical_answer,
            "context": external_answer,
            "source": external_source,
            "is_follow_up": True,
            "cache_hit": False,
            "timestamp": datetime.utcnow(),
            "used_feedback": has_feedback
        }
    else:
        chat_entry = {
            "question": req.question,
            "answer": medical_answer,
            "context": external_answer,
            "source": external_source,
            "is_follow_up": False,
            "cache_hit": False,
            "timestamp": datetime.utcnow(),
            "used_feedback": has_feedback
        }

    user_response = {
        "session_id": session_id,
        "answer": medical_answer,       # Gemini's answer (kept as requested)
        "context": external_answer,     # external RAG API answer
        "source": external_source,      # external RAG API source
        "cache_hit": False,
        "used_feedback": has_feedback
    }

    background_tasks.add_task(store_chat_in_db, session_id, chat_entry)

    return user_response

# ============================
# 🧾 Get Session History
# ============================
@app.get("/chat-history")
def get_chat_history(email: str = Query(..., description="User email"),
                    session_id: str = Query(..., description="Session ID")):
    """Get simplified chat history"""
    try:
        session_object_id = ObjectId(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid session ID format: {str(e)}")

    session = sessions_collection.find_one({"_id": session_object_id, "email": email})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or email mismatch")

    simplified_history = []
    for entry in session.get("history", []):
        if entry.get("is_follow_up", False):
            question = entry.get("original_question", "Unknown question")
        else:
            question = entry.get("question", "Unknown question")

        simplified_history.append({
            "question": question,
            "answer": entry.get("answer", "No answer"),
            "context": entry.get("context"),
            "source": entry.get("source"),
            "cache_hit": entry.get("cache_hit", False),
            "used_feedback": entry.get("used_feedback", False),
            "timestamp": entry.get("timestamp", datetime.utcnow())
        })

    return {
        "session_id": session_id,
        "email": email,
        "history": simplified_history
    }

# ============================
# 📋 Get User Sessions
# ============================
@app.get("/user-sessions")
def get_user_sessions(email: str = Query(..., description="User email")):
    """Get all sessions for a user"""
    sessions = list(sessions_collection.find(
        {"email": email},
        {"_id": 1, "created_at": 1, "history": 1}
    ).sort("created_at", -1))

    formatted_sessions = []
    for session in sessions:
        session_id_str = str(session["_id"])
        history = session.get("history", [])
        last_question = "No messages"
        if history:
            last_entry = history[-1]
            if last_entry.get("is_follow_up", False):
                last_question = last_entry.get("original_question", "Follow-up question")
            else:
                last_question = last_entry.get("question", "Unknown question")

        formatted_sessions.append({
            "session_id": session_id_str,
            "created_at": session["created_at"],
            "message_count": len(history),
            "last_question": last_question
        })

    return {
        "email": email,
        "sessions": formatted_sessions
    }

# ============================
# 🔍 Cache Management
# ============================
@app.get("/cache-stats")
async def get_cache_stats():
    """Get statistics about cached Q&A"""
    total_qa = global_qa_collection.count_documents({})
    sample_qa = list(global_qa_collection.find(
        {},
        {'question': 1, 'context': 1, 'source': 1, '_id': 0}
    ).limit(5))

    return {
        "total_cached_qa": total_qa,
        "sample_questions": sample_qa
    }

@app.delete("/cache-clear")
async def clear_cache():
    """Clear all cached Q&A"""
    result = global_qa_collection.delete_many({})
    return {"message": f"Cleared {result.deleted_count} Q&A entries from cache"}

# ============================
# 🐛 Debug Endpoint
# ============================
@app.get("/debug-session/{session_id}")
def debug_session(session_id: str):
    """Debug endpoint to see raw session data"""
    try:
        session_object_id = ObjectId(session_id)
        session = sessions_collection.find_one({"_id": session_object_id})

        if not session:
            return {"error": "Session not found"}

        session["_id"] = str(session["_id"])

        for entry in session.get("history", []):
            if "_id" in entry:
                entry["_id"] = str(entry["_id"])
            if "timestamp" in entry and isinstance(entry["timestamp"], datetime):
                entry["timestamp"] = entry["timestamp"].isoformat()

        return session
    except Exception as e:
        return {"error": f"Debug error: {str(e)}"}

# ============================
# 🚀 Startup Event
# ============================
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    print("🚀 Medical Chat API Starting Up...")
    print("✅ MongoDB Connected")
    print(f"✅ Redis Available: {REDIS_AVAILABLE}")
    print("✅ Gemini AI Configured")
    print("✅ Embedding Model Loaded")
    print("✅ Brevo Email Service Configured")
    print("✅ Feedback System Initialized")

# ============================
# 🛑 Shutdown Event
# ============================
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Medical Chat API Shutting Down...")
    client.close()
    print("✅ MongoDB Connection Closed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

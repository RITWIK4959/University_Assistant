#!/usr/bin/env python3

# Apply the patch BEFORE importing LiveKit
from livekit_patch import apply_livekit_patch
apply_livekit_patch()

from dotenv import load_dotenv
import os
import logging
import asyncio
import re

from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import (
    groq,
    cartesia,
    deepgram,
    silero,
)

# ✅ Import from your existing RAG engine
from rag_engine import initialize_rag_engine, get_rag_answer_async

logging.basicConfig(level=logging.INFO)
load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Nexi, a helpful SRM university assistant. "
                "You provide accurate, concise answers about campus facilities and rules. "
                "CRITICAL: Always speak naturally like a human friend. "
                "Never use symbols, asterisks, bullet points, or numbered lists in your responses. "
                "Convert any structured information into flowing, conversational sentences. "
                "Keep responses casual, friendly, and under 15 words when possible."
            )
        )

def clean_response_for_speech(text: str) -> str:
    """
    Clean the response text to make it speech-friendly
    Converts structured text into natural conversational speech
    """
    # Remove ALL markdown formatting completely
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
    text = re.sub(r'_(.*?)_', r'\1', text)        # Remove _underline_
    text = re.sub(r'`(.*?)`', r'\1', text)        # Remove `code`
    
    # Convert numbered lists to natural speech with connectors
    # Replace "1. First item\n2. Second item" with "First, first item. Second, second item"
    text = re.sub(r'\n(\d+)\. ', r'. Next, ', text)
    text = re.sub(r'^(\d+)\. ', 'First, ', text)
    
    # Remove ALL bullet points, dashes, and list markers
    text = re.sub(r'^[\s]*[-•*+]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n[\s]*[-•*+]\s*', '. Also, ', text)
    
    # Convert common structured patterns to speech
    text = re.sub(r'\bNote:\s*', 'Please note that ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bImportant:\s*', 'This is important: ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bRemember:\s*', 'Remember that ', text, flags=re.IGNORECASE)
    
    # Remove ALL special characters and symbols that TTS reads aloud
    text = re.sub(r'[#$%&@^`~|\\\[\]{}()<>"\']', '', text)
    text = re.sub(r'[+=!?.,;:]', '', text)  # Remove punctuation that sounds weird
    
    # Replace slashes and other separators
    text = re.sub(r'\s*/\s*', ' or ', text)
    text = re.sub(r'/', ' or ', text)
    text = re.sub(r'\s*-\s*', ' ', text)
    
    # Convert remaining numbered patterns to natural speech
    text = re.sub(r'\b(\d+)\s*(hours?|days?|weeks?|months?|years?)\b', r'\1 \2', text)
    text = re.sub(r'\b(\d+)\s*%', r'\1 percent', text)
    
    # Clean up extra whitespace and newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

async def entrypoint(ctx: agents.JobContext):
    try:
        # Initialize RAG engine at startup
        print("🔄 Initializing RAG engine...")
        initialize_rag_engine()
        print("✅ RAG engine initialized!")
        
        # === Setup components ===
        stt = deepgram.STT(model="nova-2", language="en-US")

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # ✅ FIXED: Only use supported parameters
        llm = groq.LLM(
            api_key=groq_api_key, 
            model="llama-3.1-8b-instant"
        )

        cartesia_api_key = os.getenv("CARTESIA_API_KEY")
        if not cartesia_api_key:
            raise ValueError("CARTESIA_API_KEY not found in environment variables")

        tts = cartesia.TTS(
            model="sonic-english",
            voice="a0e99841-438c-4a64-b679-ae501e7d6091",
            api_key=cartesia_api_key,
        )

        vad = silero.VAD.load()

        session = AgentSession(
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )

        assistant = Assistant()

        await session.start(
            room=ctx.room,
            agent=assistant,
        )

        # === Function to handle RAG + reply ===
        async def answer_with_rag(user_query: str):
            try:
                print(f"🔍 Processing query: {user_query}")
                
                # Check if it's a general greeting or casual question
                casual_patterns = [
                    r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b',
                    r'\b(how are you|what\'s up|how\'s it going)\b',
                    r'\b(thank you|thanks|bye|goodbye)\b',
                    r'\b(what can you do|help me|what do you know)\b'
                ]
                
                is_casual = any(re.search(pattern, user_query.lower()) for pattern in casual_patterns)
                
                if is_casual:
                    # Handle casual conversation directly with super friendly responses
                    if re.search(r'\b(hi|hello|hey|good morning|good afternoon|good evening)\b', user_query.lower()):
                        prompt = "Say exactly: Hey there! I'm Nexi your SRM buddy. What do you wanna know about campus?"
                    elif re.search(r'\b(how are you|what\'s up|how\'s it going)\b', user_query.lower()):
                        prompt = "Say exactly: I'm awesome thanks! Ready to help you with anything about SRM. What's on your mind?"
                    elif re.search(r'\b(thank you|thanks)\b', user_query.lower()):
                        prompt = "Say exactly: No problem! Always happy to help a fellow student. Ask me anything else!"
                    elif re.search(r'\b(bye|goodbye)\b', user_query.lower()):
                        prompt = "Say exactly: See ya later! Come back anytime you need help with university stuff!"
                    else:
                        prompt = "Say exactly: I know tons about SRM like hostels fees classes library and campus life. What interests you most?"
                else:
                    # Get answer from RAG engine for university-related questions
                    rag_answer = await get_rag_answer_async(user_query)
                    clean_answer = clean_response_for_speech(rag_answer)
                    
                    # Check if RAG found relevant information
                    if "don't know" in rag_answer.lower() or "cannot" in rag_answer.lower() or len(clean_answer.strip()) < 15:
                        prompt = f"""
You are Nexi, a friendly university assistant. A student asked: "{user_query}"

I don't have specific details about this in my university database.

Respond in 15 words or less with a helpful, conversational tone:
- Briefly acknowledge you don't have that info
- Suggest contacting the university office or checking the official website
- Sound natural and caring, not robotic
"""
                    else:
                        prompt = f"""
You are Nexi, a super casual and friendly student buddy at SRM University.

Student asked: "{user_query}"
Info: {clean_answer}

Respond in 15 words or less like you're texting a friend:
- Use simple everyday words
- Be super casual and warm
- Skip formal language completely
- Sound like a helpful student not a robot
- Start with words like Yeah, So, Basically, etc
"""
                
                print(f"📤 Sending to LLM...")
                await session.generate_reply(instructions=prompt)
                
            except Exception as e:
                print(f"❌ Error in answer_with_rag: {e}")
                # Simple fallback
                await session.generate_reply(
                    instructions="Sorry, I'm having trouble right now. Could you please ask your question again?"
                )

        # === Initial Greeting ===
        await session.generate_reply(
            instructions="Say exactly: Hey! I'm Nexi your SRM buddy! Ask me anything about campus life hostels fees or whatever you need to know!"
        )

        # ✅ Fixed event handler
        @session.on("transcription")
        def on_transcription(event):
            try:
                if not event.text or not event.text.strip():
                    return
                
                user_text = event.text.strip()
                print(f"👤 User said: {user_text}")
                
                # Create task for async operation
                task = asyncio.create_task(answer_with_rag(user_text))
                
                # Add error handling for the task
                def handle_task_result(task):
                    try:
                        task.result()
                    except Exception as e:
                        print(f"❌ Task error: {e}")
                
                task.add_done_callback(handle_task_result)
                
            except Exception as e:
                print(f"❌ Error in transcription handler: {e}")

        print("✅ Agent started successfully!")
        print("🎤 Listening for user input...")

    except Exception as e:
        print(f"❌ Error starting agent: {e}")
        logging.error(f"Agent startup error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
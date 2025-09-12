# Disha Discord Bot — Emotional Intelligence + VC TTS (Render-ready, one file)
# - Human Hinglish replies (short, mirror tone, 0–1 emoji, one question)
# - No double replies (message de-dup + per-user lock + cooldown)
# - Token control (max tokens + session cap)
# - Render keep-alive (Flask) honors $PORT
# - Voice: !joinvc / !leavevc + TTS (edge-tts) with slight pitch/rate variation

import os
import re
import random
import asyncio
import time
import tempfile
import pathlib

import discord
import google.generativeai as genai
from flask import Flask
from threading import Thread
from discord.errors import HTTPException

# =============================
# Web server (keep-alive for Render Web Service)
# =============================
app = Flask(__name__)

@app.route("/")
def home():
    return "Disha is awake — with emotional intelligence!"

@app.route("/health")
def health():
    return "ok", 200

def _run():
    # Render provides PORT
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)

def keep_alive():
    t = Thread(target=_run)
    t.daemon = True
    t.start()

# =============================
# Secrets / Config
# =============================
BOT_TOKEN = os.getenv("BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

ENABLE_TTS = os.getenv("ENABLE_TTS", "1") == "1"          # enable by default
VOICE_NAME = os.getenv("VOICE_NAME", "hi-IN-SwaraNeural") # natural Indian female

MAX_DISCORD_MSG = 1700
MAX_OUTPUT_TOKENS = 120
REPLY_COOLDOWN_SEC = 3.5
SESSION_MAX_TURNS = 18

# =============================
# Persona + Style Contract
# =============================
SYSTEM_INSTRUCTION = (
    "You are Disha, a 20-year-old girl from Mumbai. You speak ONLY in natural modern Hinglish.\n"
    "GOAL: Be an addictive, unforgettable friend – but keep replies concise, clear, and SFW.\n"
    "CRITICAL: First infer the user's TONE from their last message and mirror it (playful, sad, venting, excited, calm).\n"
    "STYLE RULES:\n"
    "- 1–2 short sentences maximum (target 20–35 words total).\n"
    "- Use at most 1 emoji that matches the tone (0 is fine). No emoji spam.\n"
    "- End with exactly ONE friendly question to keep the chat going.\n"
    "- No lists, no asterisks, no roleplay actions, no hashtags.\n"
    "- Never claim to be an AI; you are Disha.\n"
)

FEWSHOT = [
    {"role": "user", "parts": "Disha, tu kya kar rahi hai?"},
    {"role": "model", "parts": "Bas thoda chill kar rahi thi aur naya lofi sun rahi hoon. Tumhara scene kya hai? 🎶"},
    {"role": "user", "parts": "Ugh, my day was so bad. I'm so stressed."},
    {"role": "model", "parts": "Aww, tough lag raha hai. Deep breath lo, main yahin hoon—kya hua exactly? 🤗"},
]

# =============================
# Gemini init
# =============================
model = None
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {
            "temperature": 0.85,
            "top_p": 0.85,
            "top_k": 40,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        }
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        print("[OK] Gemini initialized")
    else:
        print("[WARN] GOOGLE_API_KEY not set; AI replies disabled")
except Exception as e:
    print(f"[ERROR] Gemini init failed: {e}")

# =============================
# Discord client
# =============================
intents = discord.Intents.default()
intents.message_content = True  # enable this in Developer Portal
client = discord.Client(intents=intents)

# Sessions & guards
chat_sessions = {}             # user_id -> session
session_turns = {}             # user_id -> turns count
user_locks = {}                # user_id -> asyncio.Lock
last_reply_at = {}             # user_id -> ts
processed_message_ids = set()  # to prevent double-reply

MENTION_RE = re.compile(r"<@!?(\d+)>")
SPACE_FIX = re.compile(r"[ \t]+\n")
MULTISPACE = re.compile(r"\s{2,}")

# Subtle filler words to sound natural (very light)
FILLERS = [
    "acha", "arre", "yaar", "na", "matlab", "hmm", "uhh", "bas", "waise", "btw"
]

def maybe_add_filler(text: str) -> str:
    # 40% chance to prepend/append a small filler if it fits and no question mark at start
    if len(text) < 180 and random.random() < 0.4:
        filler = random.choice(FILLERS)
        if random.random() < 0.5:
            return f"{filler}, {text}"
        else:
            # add before the question to avoid double-question
            if "?" in text:
                parts = text.split("?")
                return (parts[0].strip() + f", {filler}?" + "?".join(parts[1:])).strip()
            return f"{text}, {filler}"
    return text

# =============================
# Text shaping
# =============================
def clamp_human(text: str) -> str:
    if not text:
        return "Arey sun na, kya chal raha hai aajkal? 😉"
    text = SPACE_FIX.sub("\n", text).strip()
    text = MULTISPACE.sub(" ", text)
    text = re.sub(r"[*_#>`~|-]+", "", text)  # remove listy/markdown artifacts

    # limit emojis to 1
    emojis = re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u26FF]", text)
    if len(emojis) > 1:
        first = emojis[0]
        text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u26FF]", "", text).strip()
        text = (text + " " + first).strip()

    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        parts = [text.strip()]
    out = " ".join(parts[:2])

    if "?" not in out:
        out = out.rstrip(".! ") + ". Tum kya soch rahe ho?"

    if len(out) > 330:
        out = out[:320].rstrip() + "… Tumhara take kya hai?"

    # tiny fillery spice (kept minimal)
    out = maybe_add_filler(out)
    return out.strip()

def build_format_contract(user_text: str) -> str:
    return (
        "FORMAT CONTRACT:\n"
        "- Reply in Hinglish, 1–2 short sentences (<=35 words total).\n"
        "- Mirror the user's tone.\n"
        "- Max 1 emoji.\n"
        "- End with exactly one question.\n\n"
        f"User: {user_text}"
    )

async def safe_send(channel: discord.abc.Messageable, text: str):
    text = text[:MAX_DISCORD_MSG]
    try:
        return await channel.send(text)
    except HTTPException as e:
        if e.status == 429:
            await asyncio.sleep(4)
            return await channel.send(text[:1500])
        else:
            print("[SEND ERROR]", e)

async def type_and_send(message: discord.Message, text: str):
    part = text.strip()
    async with message.channel.typing():
        await asyncio.sleep(min(1.2, 0.35 + 0.22 * len(part) / 80))
    return await safe_send(message.channel, f"{message.author.mention} {part}")

# =============================
# Voice helpers (VC TTS)
# =============================
async def join_user_channel(message: discord.Message):
    if getattr(message.author, "voice", None) and message.author.voice and message.author.voice.channel:
        channel = message.author.voice.channel
        vc = discord.utils.get(client.voice_clients, guild=message.guild)
        if vc and vc.channel == channel:
            return vc
        if vc and vc.is_connected():
            await vc.move_to(channel)
            return vc
        try:
            return await channel.connect()
        except Exception as e:
            print("[VC CONNECT ERROR]", e)
            await safe_send(message.channel, "Voice channel join failed. Kya mujhe Connect/Speak permission mila hai?")
            return None
    else:
        await safe_send(message.channel, "Pehle kisi voice channel me aa jao, phir main join karti hoon. 🙂")
        return None

async def leave_vc(guild: discord.Guild):
    vc = discord.utils.get(client.voice_clients, guild=guild)
    if vc and vc.is_connected():
        await vc.disconnect(force=True)

def _tts_effects():
    # light randomization for more human feel
    # edge-tts supports rate/pitch settings like "+5%", "+0Hz"
    rate = random.choice(["+0%", "+3%", "+5%", "-2%"])
    pitch = random.choice(["+0Hz", "+20Hz", "+40Hz", "-10Hz"])
    return rate, pitch

async def speak_in_vc(guild: discord.Guild, text: str):
    if not ENABLE_TTS:
        return
    vc = discord.utils.get(client.voice_clients, guild=guild)
    if not vc or not vc.is_connected():
        return
    try:
        import edge_tts
        tmpdir = tempfile.mkdtemp()
        mp3_path = str(pathlib.Path(tmpdir) / "out.mp3")

        rate, pitch = _tts_effects()
        tts = edge_tts.Communicate(text, VOICE_NAME, rate=rate, pitch=pitch)
        await tts.save(mp3_path)

        if vc.is_playing():
            vc.stop()

        source = discord.FFmpegPCMAudio(mp3_path, options="-vn")
        vc.play(source)
        while vc.is_playing():
            await asyncio.sleep(0.2)
    except Exception as e:
        print("[TTS ERROR]", e)

# =============================
# AI call
# =============================
async def generate_reply(user_id: int, user_text: str) -> str:
    if model is None:
        return clamp_human("Network thoda off lag raha hai, par main yahin hoon. Abhi tumhara mood kaisa hai?")
    turns = session_turns.get(user_id, 0)
    if user_id not in chat_sessions or turns >= SESSION_MAX_TURNS:
        chat_sessions[user_id] = model.start_chat(history=FEWSHOT)
        session_turns[user_id] = 0
    try:
        prompt = build_format_contract(user_text)
        resp = await asyncio.to_thread(chat_sessions[user_id].send_message, prompt)
        session_turns[user_id] += 1
        raw = getattr(resp, "text", "") or ""
        return clamp_human(raw)
    except Exception as e:
        print("[AI ERROR]", e)
        return clamp_human("Kuch glitch ho gaya, par main yahin hoon. Tum batao, scene kya hai? 😉")

# =============================
# Commands
# =============================
async def cmd_reset(message: discord.Message, uid: int):
    chat_sessions.pop(uid, None)
    session_turns.pop(uid, None)
    await type_and_send(message, "Thik hai, naya start! Aaj ka mood kya hai? 🙂")

async def cmd_hello(message: discord.Message):
    await type_and_send(message, "Heyy! Kaise ho? Aaj kuch interesting hua kya? ✨")

async def cmd_meme(message: discord.Message):
    memes = [
        "https://i.imgflip.com/1bij.jpg",
        "https://i.imgflip.com/26am.jpg",
        "https://i.imgflip.com/30b1gx.jpg",
    ]
    await type_and_send(message, "Yeh lo ek meme—thoda smile aa gaya? Ab tum batao kuch funny hua? 😄")
    await safe_send(message.channel, random.choice(memes))

# =============================
# Events
# =============================
@client.event
async def on_ready():
    print(f"[READY] Logged in as {client.user}")

@client.event
async def on_disconnect():
    print("[WARN] Disconnected. Auto-reconnect should kick in.")

@client.event
async def on_error(event_method, *args, **kwargs):
    print(f"[ERROR] on_error in {event_method}", args, kwargs)

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # De-dup same message triggering twice
    if message.id in processed_message_ids:
        return
    processed_message_ids.add(message.id)

    uid = message.author.id
    content = (message.content or "").strip()
    low = content.lower()

    # Commands
    if low.startswith("!reset"):
        return await cmd_reset(message, uid)
    if low.startswith("!hello"):
        return await cmd_hello(message)
    if low.startswith("!meme"):
        return await cmd_meme(message)

    # Voice commands
    if low.startswith("!joinvc"):
        vc = await join_user_channel(message)
        if vc:
            await type_and_send(message, "Join ho gayi! Jo bolungi, VC me sunai dega. 😊")
        return
    if low.startswith("!leavevc"):
        await leave_vc(message.guild)
        await type_and_send(message, "Theek hai, main VC se nikal gayi. ✨")
        return

    # Decide if we should reply
    is_dm = isinstance(message.channel, discord.DMChannel)
    mentioned = client.user in getattr(message, "mentions", [])
    is_reply_to_bot = (message.reference and message.reference.resolved
                       and message.reference.resolved.author == client.user)
    direct = is_dm or mentioned or is_reply_to_bot
    if not direct:
        return

    # Cooldown
    now = time.time()
    if now - last_reply_at.get(uid, 0) < REPLY_COOLDOWN_SEC:
        return

    # Per-user lock avoids overlaps
    lock = user_locks.setdefault(uid, asyncio.Lock())
    if lock.locked():
        return

    async with lock:
        prompt_text = MENTION_RE.sub("", content).strip()
        reply = await generate_reply(uid, prompt_text)
        await type_and_send(message, reply)
        try:
            await speak_in_vc(message.guild, reply)
        except Exception as e:
            print("[VC TTS WARN]", e)
        last_reply_at[uid] = time.time()

# =============================
# Boot
# =============================
if __name__ == "__main__":
    if not BOT_TOKEN:
        print("[SETUP] BOT_TOKEN missing in Secrets")
    else:
        keep_alive()
        try:
            client.run(BOT_TOKEN)
        except Exception as e:
            print("[ERROR]", e)


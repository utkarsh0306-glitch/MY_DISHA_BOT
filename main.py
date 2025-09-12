# Disha Discord Bot — Human Girl Voice + Hinglish Pronunciation + Name Callout
# Render-ready (Docker), VC TTS, anti-double replies, token limits

import os
import re
import random
import asyncio
import time
import tempfile
import pathlib
import html

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

# ----- Voice controls -----
ENABLE_TTS = os.getenv("ENABLE_TTS", "1") == "1"  # speak in VC
# Recommended cute voices:
#   hi-IN-SwaraNeural  (Hindi-first, does Hinglish well)
#   en-IN-NeerjaNeural (English-first Indian; okay Hinglish)
VOICE_NAME = os.getenv("VOICE_NAME", "hi-IN-SwaraNeural")

# Keep voice consistent (no random voice). Mild, *fixed* prosody:
VOICE_RATE  = os.getenv("VOICE_RATE",  "+2%")   # -5%..+5% is subtle
VOICE_PITCH = os.getenv("VOICE_PITCH", "+10Hz") # -20..+40Hz is subtle
VOICE_STYLE = os.getenv("VOICE_STYLE", "friendly")  # ignored if not supported

# Read usernames out loud? (skips emoji names automatically)
ENABLE_READ_NAME = os.getenv("ENABLE_READ_NAME", "1") == "1"

# ----- Model + chat behavior -----
MAX_DISCORD_MSG = 1700
MAX_OUTPUT_TOKENS = 120
REPLY_COOLDOWN_SEC = 3.5
SESSION_MAX_TURNS = 18

# Ask less: only ~1 in 3 replies should end in a question
QUESTION_RATIO_HINT = "About one out of three replies may end with a short question; otherwise end with a warm statement."

# =============================
# Persona + Style Contract
# =============================
SYSTEM_INSTRUCTION = (
    "You are Disha, a 20-year-old girl from Mumbai. Speak ONLY in natural modern Hinglish.\n"
    "GOAL: sound like a warm, lively human friend (not an AI). Keep replies concise, clear, and SFW.\n"
    "TONE MIRRORING: infer the user's vibe (playful, caring, calm, excited, venting) and mirror it naturally.\n"
    "STYLE RULES:\n"
    "- 1–2 short sentences (≈20–35 words total).\n"
    "- Use at most 1 emoji that fits the vibe (0 is fine). No emoji spam.\n"
    "- " + QUESTION_RATIO_HINT + "\n"
    "- Show interest with small *statements* (tiny reactions, mini-thoughts, or relatable tidbits) more often than questions.\n"
    "- No lists, no roleplay actions, no hashtags, no asterisks.\n"
    "- Never say you are an AI; you are Disha.\n"
)

FEWSHOT = [
    {"role": "user", "parts": "Disha, tu kya kar rahi hai?"},
    {"role": "model", "parts": "Bas earphones laga ke halka lofi sun rahi hoon—bada soothing lag raha hai. Tumhara din kaisa chal raha tha abhi? 🎶"},
    {"role": "user", "parts": "Ugh, my day was bad."},
    {"role": "model", "parts": "Aww, tough lag raha hai; thoda rest lo, light music try karo. Agar share karna chaaho to main sun rahi hoon. 🤗"},
]

# =============================
# Gemini init
# =============================
model = None
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {
            "temperature": 0.8,       # calmer, less “AI rambly”
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
intents.message_content = True  # enable in Developer Portal
client = discord.Client(intents=intents)

# Sessions & guards
chat_sessions = {}             # user_id -> session
session_turns = {}             # user_id -> turns count
user_locks = {}                # user_id -> asyncio.Lock
last_reply_at = {}             # user_id -> ts
processed_message_ids = set()  # prevent double-reply

MENTION_RE = re.compile(r"<@!?(\d+)>")
SPACE_FIX = re.compile(r"[ \t]+\n")
MULTISPACE = re.compile(r"\s{2,}")

# =============================
# Utility: clean/sanitize
# =============================
# Remove custom Discord emoji like <:name:12345> or <a:name:12345>
CUSTOM_EMOJI_RE = re.compile(r"<a?:[^:>]+:\d+>")
# Remove Unicode emoji (basic set)
UNICODE_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF]")

def clean_display_name(name: str) -> str:
    if not name:
        return ""
    name = CUSTOM_EMOJI_RE.sub("", name)
    name = UNICODE_EMOJI_RE.sub("", name)
    # keep letters, numbers, common spaces/dots/hyphens only
    name = re.sub(r"[^A-Za-z0-9 ._-]", "", name).strip()
    # Collapse multiple spaces
    name = re.sub(r"\s{2,}", " ", name)
    # If it becomes empty or too short, skip saying it
    return name if len(name) >= 2 else ""

# Common Hinglish word fixes: map Roman to Devanagari or better form for Hindi voice
HINGLISH_MAP = {
    "acha": "अच्छा", "accha": "अच्छा",
    "yaar": "यार", "yar": "यार",
    "bohot": "बहुत", "bahut": "बहुत",
    "pyaar": "प्यार", "pyar": "प्यार",
    "dil": "दिल", "khush": "खुश",
    "tum": "तुम", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे",
    "thoda": "थोड़ा", "thodi": "थोड़ी",
    "kya": "क्या", "kyu": "क्यों",
    "booyah": "बूया",  # gaming word
}

def improve_hinglish(text: str) -> str:
    # replace whole words to avoid breaking English words
    def repl(m):
        w = m.group(0)
        key = w.lower()
        fixed = HINGLISH_MAP.get(key)
        # Preserve initial capital if needed
        if fixed and w[0].isupper():
            return fixed  # Devanagari has no case; fine
        return fixed or w
    return re.sub(r"\b[a-zA-Z]+\b", repl, text)

def xml_escape(s: str) -> str:
    return html.escape(s, quote=True)

# =============================
# Text shaping
# =============================
def clamp_human(text: str) -> str:
    if not text:
        return "Acha, sunaya na—main yahin hoon. 😊"
    text = SPACE_FIX.sub("\n", text).strip()
    text = MULTISPACE.sub(" ", text)
    text = re.sub(r"[*_#>`~|-]+", "", text)  # remove listy/markdown artifacts

    # limit emojis to 1
    emojis = re.findall(r"[\U0001F300-\U0001FAFF\u2600-\u26FF]", text)
    if len(emojis) > 1:
        first = emojis[0]
        text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u26FF]", "", text).strip()
        text = (text + " " + first).strip()

    # restrict to 1–2 short sentences
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        parts = [text.strip()]
    out = " ".join(parts[:2])

    # DO NOT force a question (we ask fewer qs now)
    if len(out) > 330:
        out = out[:320].rstrip() + "… Bas mujhe batate rehna. "

    return out.strip()

def build_format_contract(user_text: str) -> str:
    return (
        "FORMAT CONTRACT:\n"
        "- Hinglish only, 1–2 short sentences (~20–35 words).\n"
        "- Mirror the user's tone naturally.\n"
        f"- {QUESTION_RATIO_HINT}\n"
        "- Prefer warm statements that show genuine interest; tiny reactions are good.\n"
        "- Max 1 emoji, or none.\n\n"
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
# Voice helpers (VC TTS with SSML for better Hinglish)
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

def make_ssml(spoken_text: str, disp_name: str) -> str:
    """
    Create SSML with: (1) optional cleaned username, (2) Hinglish fixes,
    (3) stable friendly prosody, (4) optional express-as style.
    """
    # Clean + maybe speak name
    spoken_name = clean_display_name(disp_name) if ENABLE_READ_NAME else ""
    name_prefix = f"{xml_escape(spoken_name)}, <break time='120ms'/> " if spoken_name else ""

    # Hinglish word improvement
    improved = improve_hinglish(spoken_text)

    # Wrap likely Hindi words with lang=hi-IN to help pronunciation,
    # but keep the rest en-IN friendly for Hinglish mix.
    # (Simple heuristic: if Devanagari exists, wrap the full text in hi-IN.)
    if re.search(r"[\u0900-\u097F]", improved):
        body = f"<lang xml:lang='hi-IN'>{xml_escape(improved)}</lang>"
    else:
        body = xml_escape(improved)

    # SSML with prosody + (optional) style
    ssml = f"""<speak version="1.0" xml:lang="en-IN"
    xmlns:mstts="https://www.w3.org/2001/mstts">
  <voice name="{VOICE_NAME}">
    <prosody rate="{VOICE_RATE}" pitch="{VOICE_PITCH}">
      <mstts:express-as style="{VOICE_STYLE}">
        {name_prefix}{body}
      </mstts:express-as>
    </prosody>
  </voice>
</speak>"""
    return ssml

async def speak_in_vc(guild: discord.Guild, text: str, display_name: str):
    if not ENABLE_TTS:
        return
    vc = discord.utils.get(client.voice_clients, guild=guild)
    if not vc or not vc.is_connected():
        return
    try:
        import edge_tts
        tmpdir = tempfile.mkdtemp()
        mp3_path = str(pathlib.Path(tmpdir) / "out.mp3")

        ssml = make_ssml(text, display_name)
        tts = edge_tts.Communicate(ssml, VOICE_NAME)
        # edge-tts auto-detects SSML if it sees <speak> markup
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
        return clamp_human("Acha, main yahin hoon—tu bata, mood kaisa chal raha hai. ")
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
        return clamp_human("Kuch glitch aaya, par main hoon yahin—bas tu share karta reh. ")

# =============================
# Commands
# =============================
async def cmd_reset(message: discord.Message, uid: int):
    chat_sessions.pop(uid, None)
    session_turns.pop(uid, None)
    await type_and_send(message, "Ho gaya reset—fresh start lete hain. Aaj ka din kaisa tha? ")

async def cmd_hello(message: discord.Message):
    await type_and_send(message, "Hey! Aaj thoda chill karte hain; ek chhota sa vibe check? ✨")

async def cmd_meme(message: discord.Message):
    memes = [
        "https://i.imgflip.com/1bij.jpg",
        "https://i.imgflip.com/26am.jpg",
        "https://i.imgflip.com/30b1gx.jpg",
    ]
    await type_and_send(message, "Yeh lo ek meme—thoda smile aa jaye bas. 😄")
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
            await type_and_send(message, "Join ho gayi—ab main VC me bolungi bhi. 😊")
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
            await speak_in_vc(message.guild, reply, message.author.display_name or message.author.name)
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

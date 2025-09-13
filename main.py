# Disha Discord Bot — human vibe + VC TTS
# Fixes:
# - Strong single-reply guard (no doubles)
# - TTS sanitization (won't read code/mentions/links)
# - Cute stable voice + better Hinglish + name callout

import os
import re
import random
import asyncio
import time
import tempfile
import pathlib
import html
from collections import deque

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
ENABLE_TTS = os.getenv("ENABLE_TTS", "1") == "1"
VOICE_NAME = os.getenv("VOICE_NAME", "hi-IN-SwaraNeural")  # try en-IN-NeerjaNeural too
VOICE_RATE  = os.getenv("VOICE_RATE",  "+2%")
VOICE_PITCH = os.getenv("VOICE_PITCH", "+10Hz")
VOICE_STYLE = os.getenv("VOICE_STYLE", "friendly")
ENABLE_READ_NAME = os.getenv("ENABLE_READ_NAME", "1") == "1"

# ----- Model + chat behavior -----
MAX_DISCORD_MSG = 1700
MAX_OUTPUT_TOKENS = 120
REPLY_COOLDOWN_SEC = 3.5
SESSION_MAX_TURNS = 18
QUESTION_RATIO_HINT = "About one out of three replies may end with a short question; otherwise end with a warm statement."

# For diagnosing duplicates
INSTANCE_ID = os.getenv("RENDER_INSTANCE_ID") or os.getenv("HOSTNAME") or str(os.getpid())

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
    "- Prefer warm statements that show genuine interest over constant questions.\n"
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
            "temperature": 0.8,
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
intents.message_content = True
client = discord.Client(intents=intents)

# Sessions & guards
chat_sessions = {}             # user_id -> session
session_turns = {}             # user_id -> turns count
user_locks = {}                # user_id -> asyncio.Lock
last_reply_at = {}             # user_id -> ts

# --- single-reply guard across this process + avoid repeating same sentence
_PROCESSED_IDS = set()
_PROCESSED_ORDER = deque(maxlen=10000)
_LAST_REPLY_NORM = {}  # user_id -> last normalized reply

def already_processed(mid: int) -> bool:
    if mid in _PROCESSED_IDS:
        return True
    _PROCESSED_IDS.add(mid)
    _PROCESSED_ORDER.append(mid)
    if len(_PROCESSED_ORDER) == _PROCESSED_ORDER.maxlen:
        old = _PROCESSED_ORDER.popleft()
        _PROCESSED_IDS.discard(old)
    return False

def _norm_reply(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return re.sub(r"[!? .]+$", "", s)

MENTION_RE = re.compile(r"<@!?(\d+)>")
SPACE_FIX = re.compile(r"[ \t]+\n")
MULTISPACE = re.compile(r"\s{2,}")

# =============================
# TTS sanitization (so it won't read code/mentions/links)
# =============================
# Mentions/tags
MENTION_TAG_RE   = re.compile(r"<@!?[0-9]+>")     # user mention
ROLE_TAG_RE      = re.compile(r"<@&[0-9]+>")      # role mention
CHANNEL_TAG_RE   = re.compile(r"<#[0-9]+>")       # channel mention
TIMESTAMP_TAG_RE = re.compile(r"<t:[0-9]+(?::[a-zA-Z])?>")
CUSTOM_EMOJI_RE  = re.compile(r"<a?:[^:>]+:\d+>") # <:name:123>, <a:name:123>

# Links / code
URL_RE           = re.compile(r"https?://\S+")
MD_LINK_RE       = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [text](url) -> text
CODE_FENCE_RE    = re.compile(r"```.*?```", re.DOTALL)   # triple-backtick block
INLINE_CODE_RE   = re.compile(r"`[^`]*`")                # inline code

# Misc markdown/symbol cleanup for speech
MARKDOWN_SYM_RE  = re.compile(r"[*_~`|>]+")
AT_SIGN_RE       = re.compile(r"@")  # leftover at-signs

def tts_sanitize(text: str) -> str:
    # remove big blocks first
    text = CODE_FENCE_RE.sub(" ", text)
    # strip mentions/tags/custom emoji
    for rx in (MENTION_TAG_RE, ROLE_TAG_RE, CHANNEL_TAG_RE, TIMESTAMP_TAG_RE, CUSTOM_EMOJI_RE):
        text = rx.sub(" ", text)
    # links -> drop, markdown links -> keep visible text
    text = URL_RE.sub(" ", text)
    text = MD_LINK_RE.sub(r"\1", text)
    # drop inline code/backticks
    text = INLINE_CODE_RE.sub(" ", text)
    # clean markdown artifacts
    text = MARKDOWN_SYM_RE.sub(" ", text)
    text = AT_SIGN_RE.sub(" ", text)
    # collapse whitespace / trim
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =============================
# Name cleanup & Hinglish fixes
# =============================
UNICODE_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF]")

def clean_display_name(name: str) -> str:
    if not name:
        return ""
    name = CUSTOM_EMOJI_RE.sub("", name)
    name = UNICODE_EMOJI_RE.sub("", name)
    name = re.sub(r"[^A-Za-z0-9 ._-]", "", name).strip()
    name = re.sub(r"\s{2,}", " ", name)
    return name if len(name) >= 2 else ""

HINGLISH_MAP = {
    "acha": "अच्छा", "accha": "अच्छा",
    "yaar": "यार", "yar": "यार",
    "bohot": "बहुत", "bahut": "बहुत",
    "pyaar": "प्यार", "pyar": "प्यार",
    "dil": "दिल", "khush": "खुश",
    "tum": "तुम", "mera": "मेरा", "meri": "मेरी", "mere": "मेरे",
    "thoda": "थोड़ा", "thodi": "थोड़ी",
    "kya": "क्या", "kyu": "क्यों",
    "mast": "मस्त", "scene": "सीन",
    "booyah": "बूया",
}

def improve_hinglish(text: str) -> str:
    def repl(m):
        w = m.group(0)
        key = w.lower()
        fixed = HINGLISH_MAP.get(key)
        if fixed and w[0].isupper():
            return fixed
        return fixed or w
    return re.sub(r"\b[a-zA-Z]+\b", repl, text)

def xml_escape(s: str) -> str:
    return html.escape(s, quote=True)

# =============================
# Text shaping
# =============================
def clamp_human(text: str) -> str:
    if not text:
        return "Acha, main yahin hoon—tu bata, mood kaisa chal raha hai. "
    text = SPACE_FIX.sub("\n", text).strip()
    text = MULTISPACE.sub(" ", text)
    text = re.sub(r"[*_#>`~|-]+", "", text)

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

    if len(out) > 330:
        out = out[:320].rstrip() + "… Bas mujhe batate rehna. "
    return out.strip()

def build_format_contract(user_text: str) -> str:
    return (
        "FORMAT CONTRACT:\n"
        "- Hinglish only, 1–2 short sentences (~20–35 words).\n"
        "- Mirror the user's tone naturally.\n"
        "- Ask a question at MOST 1 out of 3 replies; otherwise end with a warm statement.\n"
        "- Show interest with tiny reactions or relatable thoughts.\n"
        "- Max 1 emoji, or none.\n"
        "- Do NOT include @mentions, hashtags, links, code, or commands in your reply.\n\n"
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
# Voice helpers (VC TTS with SSML)
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
    # Sanitize first so TTS never reads code/mentions/links
    spoken_text = tts_sanitize(spoken_text)

    # Optional: speak a cleaned username
    spoken_name = clean_display_name(disp_name) if ENABLE_READ_NAME else ""
    name_prefix = f"{xml_escape(spoken_name)}, <break time='120ms'/> " if spoken_name else ""

    # Hinglish improvements
    improved = improve_hinglish(spoken_text)

    # If Devanagari present, hint Hindi pronunciation
    if re.search(r"[\u0900-\u097F]", improved):
        body = f"<lang xml:lang='hi-IN'>{xml_escape(improved)}</lang>"
    else:
        body = xml_escape(improved)

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

# Extra: show which instance is replying (diagnose doubles)
async def cmd_who(message: discord.Message):
    await type_and_send(message, f"Instance: `{INSTANCE_ID}`")

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

    # Single-reply guard for this process
    if already_processed(message.id):
        return

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
    if low.startswith("!who"):
        return await cmd_who(message)

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

        # Don't send the exact same line twice to this user
        nr = _norm_reply(reply)
        if _LAST_REPLY_NORM.get(uid) == nr:
            reply = re.sub(r"[!? .]+$", "", reply).strip() + " — theek lag raha hai, main saath hoon. 🙂"
        _LAST_REPLY_NORM[uid] = _norm_reply(reply)

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

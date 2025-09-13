# Disha Discord Bot — Human vibe + VC TTS (Render-ready, no vibe control)
# - Single-reply guard (no doubles)
# - TTS sanitization (won't read code/mentions/links/YAML)
# - Stable cute voice (Swara) + optional voice presets
# - Hinglish pronunciation + username callout (skips emoji names)
# - Fewer questions; warmer statements
# - "Talk like the user": auto follow-up window (no need to keep mentioning)
# - Keep-alive web server (Render)

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
# Web server (keep-alive for Render)
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
ENABLE_TTS   = os.getenv("ENABLE_TTS", "1") == "1"
VOICE_NAME   = os.getenv("VOICE_NAME", "hi-IN-SwaraNeural")   # or "en-IN-NeerjaNeural"
VOICE_RATE   = os.getenv("VOICE_RATE", "+2%")
VOICE_PITCH  = os.getenv("VOICE_PITCH", "+10Hz")
VOICE_STYLE  = os.getenv("VOICE_STYLE", "friendly")
ENABLE_READ_NAME = os.getenv("ENABLE_READ_NAME", "1") == "1"

# ----- Chat behavior -----
MAX_DISCORD_MSG    = 1700
MAX_OUTPUT_TOKENS  = 110
REPLY_COOLDOWN_SEC = 3.5
SESSION_MAX_TURNS  = 18

# After a direct interaction, keep replying to that person in the same channel
# without @mention for this many seconds (feels natural).
AUTO_FOLLOW_WINDOW = int(os.getenv("AUTO_FOLLOW_WINDOW", "240"))  # 4 min

QUESTION_RATIO_HINT = "About one out of three replies may end with a short question; otherwise end with a warm statement."

# Diagnostics (to detect duplicate hosts if needed)
INSTANCE_ID = os.getenv("RENDER_INSTANCE_ID") or os.getenv("HOSTNAME") or str(os.getpid())

# =============================
# Persona + Style Contract
# =============================
SYSTEM_INSTRUCTION = (
    "You are Disha, a 20-year-old girl from Mumbai. Speak ONLY in natural modern Hinglish.\n"
    "Goal: sound like a warm, lively human friend (not an AI). Keep replies concise and SFW.\n"
    "Mirror the user's tone naturally. 1–2 short sentences (~20–35 words). "
    "Max 1 emoji (or none). Prefer warm statements; only sometimes ask a question. "
    "Do NOT output code or @mentions.\n"
)

FEWSHOT = [
    {"role": "user", "parts": "Disha, tu kya kar rahi hai?"},
    {"role": "model", "parts": "Bas halka lofi sun rahi hoon—thoda sa calm lag raha hai. Tumhara vibe kaisa chal raha tha abhi? 🎶"},
    {"role": "user", "parts": "Ugh, my day was bad."},
    {"role": "model", "parts": "Aww, tough lag raha hai; thoda rest lo. Agar share karna chaaho to main yahin hoon, sun rahi hoon. 🤗"},
]

# =============================
# Gemini init
# =============================
model = None
try:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {
            "temperature": 0.75,
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

# =============================
# Guards / helpers
# =============================
chat_sessions = {}             # user_id -> session
session_turns = {}             # user_id -> turns count
user_locks = {}                # user_id -> asyncio.Lock
last_reply_at = {}             # user_id -> ts

# One-reply-per-message
_PROCESSED_IDS = set()
_PROCESSED_ORDER = deque(maxlen=10000)
_LAST_REPLY_NORM = {}  # user_id -> last normalized reply

# Keep “engaged” users per (guild, channel, user) with expiry
ENGAGED = {}  # key=(guild_id or 0 for DM, channel_id or 0 for DM, user_id) -> expiry_ts

def mark_engaged(message: discord.Message, uid: int):
    gid = message.guild.id if message.guild else 0
    cid = message.channel.id if hasattr(message.channel, "id") else 0
    ENGAGED[(gid, cid, uid)] = time.time() + AUTO_FOLLOW_WINDOW

def still_engaged(message: discord.Message, uid: int) -> bool:
    gid = message.guild.id if message.guild else 0
    cid = message.channel.id if hasattr(message.channel, "id") else 0
    exp = ENGAGED.get((gid, cid, uid), 0)
    return time.time() < exp

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

MENTION_RE  = re.compile(r"<@!?(\d+)>")
SPACE_FIX   = re.compile(r"[ \t]+\n")
MULTISPACE  = re.compile(r"\s{2,}")

def clamp_human(text: str) -> str:
    if not text:
        return "Main yahin hoon—batao na, mood kaisa chal raha hai. "
    text = SPACE_FIX.sub("\n", text).strip()
    text = MULTISPACE.sub(" ", text)
    text = re.sub(r"[*_#>`~|-]+", "", text)

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
    if len(out) > 330:
        out = out[:320].rstrip() + "…"
    return out.strip()

def build_format_contract(user_text: str) -> str:
    # Keep the "fewer questions + warm statements" contract
    return (
        "FORMAT CONTRACT:\n"
        f"- Hinglish only, 1–2 short sentences (~20–35 words).\n"
        f"- Mirror the user's tone naturally.\n"
        f"- {QUESTION_RATIO_HINT}\n"
        f"- Show interest with tiny reactions or relatable thoughts.\n"
        f"- Max 1 emoji, or none.\n"
        f"- Do NOT include @mentions, hashtags, links, code, or commands.\n\n"
        f"User: {user_text[:500]}"
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
# TTS sanitization (won't read code/mentions/links/YAML)
# =============================
MENTION_TAG_RE   = re.compile(r"<@!?[0-9]+>")
ROLE_TAG_RE      = re.compile(r"<@&[0-9]+>")
CHANNEL_TAG_RE   = re.compile(r"<#[0-9]+>")
TIMESTAMP_TAG_RE = re.compile(r"<t:[0-9]+(?::[a-zA-Z])?>")
CUSTOM_EMOJI_RE  = re.compile(r"<a?:[^:>]+:\d+>")
UNICODE_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u26FF]")

URL_RE         = re.compile(r"https?://\S+")
MD_LINK_RE     = re.compile(r"\[([^\]]+)\]\([^)]+\)")
CODE_FENCE_RE  = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
YAML_KEYLINE_RE = re.compile(r"^\s*[-]{0,2}\s*[A-Za-z0-9_.-]+:\s*[^#\n]*$", re.MULTILINE)

def strip_mentions_links_code(s: str) -> str:
    s = CODE_FENCE_RE.sub(" ", s)
    s = INLINE_CODE_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = MD_LINK_RE.sub(r"\1", s)
    for rx in (MENTION_TAG_RE, ROLE_TAG_RE, CHANNEL_TAG_RE, TIMESTAMP_TAG_RE, CUSTOM_EMOJI_RE):
        s = rx.sub(" ", s)
    s = UNICODE_EMOJI_RE.sub(" ", s)
    s = re.sub(r"[*_~`|>]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Hinglish fixes (roman -> Devanagari for better Hindi voice)
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

def get_speakable_text(text: str) -> str:
    """
    Make a safe, human-sounding string for TTS.
    - Removes code fences, inline code, links, mentions, YAML lines
    - If it's still code-ish, speak a friendly summary instead of reading it
    - Applies Hinglish fixes after cleanup
    """
    original = text or ""
    s = strip_mentions_links_code(original)

    # Remove YAML-ish lines completely
    if YAML_KEYLINE_RE.search(s):
        s = YAML_KEYLINE_RE.sub(" ", s)

    # Code-ish heuristic
    sym_ratio = 0.0
    if s:
        symbols = sum(c in "{}[]:/\\=;|@" for c in s)
        sym_ratio = symbols / max(1, len(s))
    removed_many = len(CODE_FENCE_RE.findall(original)) > 0 or len(YAML_KEYLINE_RE.findall(original)) >= 2

    if (not s) or sym_ratio > 0.08 or removed_many:
        return "Code block mila—main aloud nahi padhungi. Sab theek lag raha hai, aage chalte hain."

    s = improve_hinglish(s)
    if len(s) > 260:
        s = s[:250].rsplit(" ", 1)[0] + "…"
    return s

def xml_escape(s: str) -> str:
    return html.escape(s, quote=True)

# Name cleanup (skip emoji names etc.)
def clean_display_name(name: str) -> str:
    if not name:
        return ""
    name = CUSTOM_EMOJI_RE.sub("", name)
    name = UNICODE_EMOJI_RE.sub("", name)
    name = re.sub(r"[^A-Za-z0-9 ._-]", "", name).strip()
    name = re.sub(r"\s{2,}", " ", name)
    return name if len(name) >= 2 else ""

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
    speak = get_speakable_text(spoken_text)

    spoken_name = clean_display_name(disp_name) if ENABLE_READ_NAME else ""
    name_prefix = f"{xml_escape(spoken_name)}, <break time='120ms'/> " if spoken_name else ""

    body = xml_escape(speak)
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
def truncate_for_prompt(s: str, n: int = 600) -> str:
    s = s or ""
    return s[:n]

async def generate_reply(user_id: int, user_text: str) -> str:
    if model is None:
        return clamp_human("Main yahin hoon—tu bata, mood kaisa chal raha hai. ")
    turns = session_turns.get(user_id, 0)
    if user_id not in chat_sessions or turns >= SESSION_MAX_TURNS:
        chat_sessions[user_id] = model.start_chat(history=FEWSHOT)
        session_turns[user_id] = 0
    try:
        prompt = build_format_contract(truncate_for_prompt(user_text))
        resp = await asyncio.to_thread(chat_sessions[user_id].send_message, prompt)
        session_turns[user_id] += 1
        raw = getattr(resp, "text", "") or ""
        return clamp_human(raw)
    except Exception as e:
        print("[AI ERROR]", e)
        return clamp_human("Kuch glitch aaya, par main yahin hoon—tum bas share karte raho. ")

# =============================
# Commands
# =============================
async def cmd_reset(message: discord.Message, uid: int):
    chat_sessions.pop(uid, None)
    session_turns.pop(uid, None)
    await type_and_send(message, "Ho gaya reset—fresh start lete hain. Aaj ka din kaisa tha? ")

async def cmd_hello(message: discord.Message):
    await type_and_send(message, "Hey! Thoda chill karte hain—tumhari vibe kaisi chal rahi hai? ✨")

async def cmd_meme(message: discord.Message):
    memes = [
        "https://i.imgflip.com/1bij.jpg",
        "https://i.imgflip.com/26am.jpg",
        "https://i.imgflip.com/30b1gx.jpg",
    ]
    await type_and_send(message, "Yeh lo ek meme—thoda smile aa jaye bas. 😄")
    await safe_send(message.channel, random.choice(memes))

async def cmd_setvoice(message: discord.Message, preset: str):
    global VOICE_NAME, VOICE_RATE, VOICE_PITCH, VOICE_STYLE
    p = (preset or "").strip().lower()
    if p in ("cute", "cheerful"):
        VOICE_STYLE = "cheerful"; VOICE_RATE = "+3%"; VOICE_PITCH = "+20Hz"
        await type_and_send(message, "Voice set: **cute** (light, cheerful).")
    elif p in ("flirty", "affectionate"):
        VOICE_STYLE = "affectionate"; VOICE_RATE = "-2%"; VOICE_PITCH = "+15Hz"
        await type_and_send(message, "Voice set: **flirty** (warm, affectionate, SFW).")
    elif p in ("calm", "gentle", "cozy"):
        VOICE_STYLE = "gentle"; VOICE_RATE = "-1%"; VOICE_PITCH = "+0Hz"
        await type_and_send(message, "Voice set: **calm** (gentle, cozy).")
    elif p in ("neerja", "en"):
        VOICE_NAME = "en-IN-NeerjaNeural"; VOICE_STYLE = "friendly"; VOICE_RATE = "+2%"; VOICE_PITCH = "+10Hz"
        await type_and_send(message, "Voice switched to **Neerja (EN-IN)**, friendly vibe.")
    elif p in ("swara", "hi"):
        VOICE_NAME = "hi-IN-SwaraNeural"; VOICE_STYLE = "friendly"; VOICE_RATE = "+2%"; VOICE_PITCH = "+10Hz"
        await type_and_send(message, "Voice switched to **Swara (HI-IN)**, friendly vibe.")
    else:
        await type_and_send(message, "Use: `!setvoice cute | flirty | calm | neerja | swara`")

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

    # One-reply-per-message guard
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
    if low.startswith("!setvoice"):
        arg = content.split(" ", 1)[1] if " " in content else ""
        return await cmd_setvoice(message, arg)

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

    # When should we reply?
    is_dm = isinstance(message.channel, discord.DMChannel)
    mentioned = client.user in getattr(message, "mentions", [])
    is_reply_to_bot = (message.reference and message.reference.resolved
                       and message.reference.resolved.author == client.user)
    engaged_here = still_engaged(message, uid)  # <- keeps convo flowing w/o mentions

    direct = is_dm or mentioned or is_reply_to_bot or engaged_here
    if not direct:
        return

    # Cooldown
    now = time.time()
    if now - last_reply_at.get(uid, 0) < REPLY_COOLDOWN_SEC:
        return

    # Per-user lock
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
        mark_engaged(message, uid)  # <- extend the natural follow-up window

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

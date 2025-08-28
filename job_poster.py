import os
import json
import time
import logging
import argparse
from typing import List, Optional
from dotenv import load_dotenv
from datetime import datetime

import requests
import schedule

import telebot
from telebot import types
load_dotenv()
# Try import for Gemini official client
GEMINI_CLIENT_AVAILABLE = False
GENAI_CLIENT_AVAILABLE = False
try:
    import google.generativeai as gia  # type: ignore
    GEMINI_CLIENT_AVAILABLE = True
except Exception:
    pass

try:
    # Newer client style
    from google import genai  # type: ignore
    GENAI_CLIENT_AVAILABLE = True
except Exception:
    GENAI_CLIENT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

POSTED_FILE = os.path.join(os.path.dirname(__file__), "posted.json")

# Cached MongoDB collection handle (initialized on first use if configured)
_mongo_collection = None


def get_mongo_collection():
    """Return a MongoDB collection for posted IDs if configured and available, else None.

    Uses env vars:
      - MONGODB_URI (enables MongoDB mode when present)
      - MONGODB_DB (default: jobposter)
      - MONGODB_COLLECTION (default: posted_jobs)
    """
    global _mongo_collection

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        return None

    if _mongo_collection is not None:
        return _mongo_collection

    try:
        # Import locally so environments without Mongo can still run with file fallback
        from pymongo import MongoClient
    except Exception:
        logging.warning("MONGODB_URI provided but pymongo is not installed. Falling back to JSON file storage.")
        return None

    db_name = os.environ.get("MONGODB_DB", "jobposter")
    coll_name = os.environ.get("MONGODB_COLLECTION", "posted_jobs")

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Verify connection early
        client.admin.command("ping")
        _mongo_collection = client[db_name][coll_name]
        logging.info("MongoDB storage enabled for posted jobs: db=%s collection=%s", db_name, coll_name)
        return _mongo_collection
    except Exception as e:
        logging.error("Failed to connect to MongoDB. Falling back to JSON file storage. Error: %s", e)
        _mongo_collection = None
        return None


def load_posted_ids() -> List[int]:
    # Prefer MongoDB if configured and available
    coll = get_mongo_collection()
    if coll is not None:
        try:
            ids = [doc.get("_id") for doc in coll.find({}, {"_id": 1})]
            # Ensure ints for consistency with existing code
            out: List[int] = []
            for v in ids:
                try:
                    out.append(int(v))
                except Exception:
                    continue
            return out
        except Exception as e:
            logging.error("Failed to load posted IDs from MongoDB: %s", e)
            # Fall through to file storage

    if not os.path.exists(POSTED_FILE):
        return []
    try:
        with open(POSTED_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed to load posted IDs from file: %s", e)
        return []


def save_posted_ids(ids: List[int]):
    # If MongoDB is enabled, insert only missing IDs as documents { _id: <post_id>, postedAt: <utc> }
    coll = get_mongo_collection()
    if coll is not None:
        try:
            existing = set(doc.get("_id") for doc in coll.find({}, {"_id": 1}))
            to_insert = []
            for pid in ids:
                if pid not in existing:
                    to_insert.append({"_id": int(pid), "postedAt": datetime.utcnow()})
            if to_insert:
                coll.insert_many(to_insert, ordered=False)
        except Exception as e:
            logging.error("Failed to save posted IDs to MongoDB: %s", e)
            # As safety, also try file write below
        else:
            # If Mongo path succeeded, also keep the file in sync for local debugging
            try:
                with open(POSTED_FILE, "w", encoding="utf-8") as f:
                    json.dump(sorted(set(ids)), f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return

    # Fallback to file-based storage
    try:
        with open(POSTED_FILE, "w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error("Failed to save posted IDs to file: %s", e)


def fetch_posts(use_sample: bool = False) -> List[dict]:
    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "res.json")
        logging.info("Loading sample posts from %s", sample_path)
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = "https://urjii-jobs.com/wp-json/wp/v2/posts?_embed"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error("Failed to fetch posts: %s", e)
        return []


def extract_post_fields(post: dict) -> dict:
    post_id = post.get("id")
    title = post.get("title", {}).get("rendered", "").strip()
    link = post.get("link")
    content = post.get("content", {}).get("rendered", "").strip()

    # Try to find featured image via _embedded
    featured_url = None
    try:
        embedded = post.get("_embedded", {})
        media = embedded.get("wp:featuredmedia")
        if media and isinstance(media, list) and len(media) > 0:
            featured_url = media[0].get("source_url")
    except Exception:
        featured_url = None

    return {
        "id": post_id,
        "title": title,
        "link": link,
        "content": content,
        "featured_image": featured_url,
        "raw": post,
    }


def extract_tags_from_post(post: dict) -> List[str]:
    """Attempt to extract up to two tag names from the post.
    Prefer embedded terms (wp:term). If not available, fall back to any provided tag names in the raw object.
    Returns a list of tag strings (no '#').
    """
    names: List[str] = []
    try:
        raw = post.get("raw") or {}
        # Try embedded terms
        embedded = raw.get("_embedded") or {}
        wp_term = embedded.get("wp:term")
        if wp_term and isinstance(wp_term, list):
            # wp:term is list of term-lists grouped by taxonomy; look for post_tag
            for term_group in wp_term:
                if isinstance(term_group, list) and len(term_group) > 0:
                    # check first item's taxonomy
                    first = term_group[0]
                    if first.get("taxonomy") == "post_tag":
                        for t in term_group:
                            if len(names) >= 2:
                                break
                            n = t.get("name")
                            if n:
                                names.append(n)
                        if names:
                            break

        # Fallback: some APIs include 'tags' as list of names (rare)
        if not names:
            tags_field = raw.get("tags") or raw.get("tag")
            if isinstance(tags_field, list) and tags_field:
                # if tags are names (strings)
                if all(isinstance(x, str) for x in tags_field):
                    names = tags_field[:2]

        # Final fallback: nothing found -> empty list
    except Exception:
        names = []

    return names[:2]


def configure_gemini(api_key: str):
    configured = False
    if GEMINI_CLIENT_AVAILABLE:
        try:
            gia.configure(api_key=api_key)
            configured = True
        except Exception as e:
            logging.error("Failed to configure google.generativeai client: %s", e)

    if GENAI_CLIENT_AVAILABLE:
        try:
            # genai.Client can accept api_key in constructor for some versions
            # We'll store a client on the module for later use
            try:
                genai_client = genai.Client(api_key=api_key)
            except Exception:
                genai_client = genai.Client()
            # attach to module global
            globals()["genai_client"] = genai_client
            configured = True
        except Exception as e:
            logging.error("Failed to configure genai client: %s", e)

    if not configured:
        logging.warning("No Gemini/GenAI client configured. Summarization will use local fallback.")


def summarize_with_gemini(text: str, max_chars: int, model: str = "gemini-2.1") -> Optional[str]:
    # Build clear instruction for the model
    prompt = (
        f"You are a copywriter. Summarize the following job posting to at most {max_chars} characters. "
        "Do NOT include any application links, URLs, or contact details in the summary. "
        "Make the summary attractive and action-driving so readers will click the 'Apply Here' button. add some informations like salary , roles or requirements if there "
        "Return plain HTML-safe text (no surrounding <html> tag). Only provide the summary body — do not add tags, title, or signatures."
        "list with  👉 when listing is needed in the text like requirements"
        "\n\n" + text
    )

    def debug_dump_response(resp):
        try:
            logging.error("--- Raw response debug start ---")
            logging.error("type: %s", type(resp))
            try:
                logging.error("repr: %s", repr(resp))
            except Exception:
                pass
            try:
                # attempt to convert to dict
                if isinstance(resp, dict):
                    logging.error("dict keys: %s", list(resp.keys()))
                    logging.error("json: %s", json.dumps(resp, default=str)[:2000])
                else:
                    d = getattr(resp, "__dict__", None)
                    if d:
                        logging.error("__dict__ keys: %s", list(d.keys()))
                        logging.error("__dict__ sample: %s", json.dumps({k: str(v)[:1000] for k, v in d.items()}, default=str)[:2000])
            except Exception:
                pass
            try:
                logging.error("dir(resp): %s", [x for x in dir(resp) if not x.startswith("__")][:200])
            except Exception:
                pass
            logging.error("--- Raw response debug end ---")
        except Exception:
            pass

    def generic_extract_text(resp) -> Optional[str]:
        # Try several common attribute/key names, and if resp is nested dict/list, search recursively for the longest string
        candidates = []
        try:
            # attr names to check
            for attr in ("text", "output_text", "content", "response", "result", "answer", "output"):
                val = getattr(resp, attr, None)
                if val and isinstance(val, str):
                    candidates.append(val)
                if val and isinstance(val, list) and len(val) > 0:
                    # try to pull strings from list
                    for it in val:
                        if isinstance(it, str):
                            candidates.append(it)
                        elif isinstance(it, dict):
                            for v in it.values():
                                if isinstance(v, str):
                                    candidates.append(v)

            # If resp is dict-like
            if isinstance(resp, dict):
                def walk(o):
                    if isinstance(o, str):
                        candidates.append(o)
                    elif isinstance(o, dict):
                        for v in o.values():
                            walk(v)
                    elif isinstance(o, list):
                        for i in o:
                            walk(i)
                walk(resp)

            # If object with __dict__
            d = getattr(resp, "__dict__", None)
            if d:
                for v in d.values():
                    if isinstance(v, str):
                        candidates.append(v)
                    elif isinstance(v, (list, dict)):
                        try:
                            if isinstance(v, dict):
                                for vv in v.values():
                                    if isinstance(vv, str):
                                        candidates.append(vv)
                            else:
                                for vv in v:
                                    if isinstance(vv, str):
                                        candidates.append(vv)
                        except Exception:
                            pass

        except Exception:
            pass

        # select the longest candidate (likely the main text)
        if candidates:
            candidates = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
            if not candidates:
                return None
            candidates.sort(key=lambda s: len(s), reverse=True)
            return candidates[0][:max_chars]
        return None

    # Try genai client first (newer client)
    try:
        if GENAI_CLIENT_AVAILABLE and globals().get("genai_client"):
            client = globals().get("genai_client")
            try:
                resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                # Try direct common fields
                text_out = getattr(resp, "text", None) or getattr(resp, "response", None) or getattr(resp, "content", None)
                if isinstance(text_out, str) and text_out:
                    return text_out[:max_chars]

                # look for candidates or nested fields
                candidates = getattr(resp, "candidates", None) or getattr(resp, "outputs", None) or getattr(resp, "choices", None)
                if candidates:
                    # try different shapes
                    try:
                        # if list of dicts
                        if isinstance(candidates, list) and len(candidates) > 0:
                            first = candidates[0]
                            if isinstance(first, dict):
                                for key in ("content", "text", "output", "message"):
                                    val = first.get(key)
                                    if isinstance(val, str) and val:
                                        return val[:max_chars]
                                    if isinstance(val, list) and len(val) > 0:
                                        # pick first text-like
                                        for item in val:
                                            if isinstance(item, str):
                                                return item[:max_chars]
                                            if isinstance(item, dict):
                                                for vv in item.values():
                                                    if isinstance(vv, str):
                                                        return vv[:max_chars]
                    except Exception:
                        pass

                # As a last effort try generic extraction
                got = generic_extract_text(resp)
                if got:
                    return got

                # If we reach here, dump debug info for developer
                logging.error("genai client returned an unexpected response structure. Dumping response for debugging.")
                debug_dump_response(resp)
            except Exception as e:
                logging.error("genai client call failed: %s", e)

        # Try older google.generativeai client shapes if available
        if GEMINI_CLIENT_AVAILABLE:
            try:
                if hasattr(gia, "generate_text"):
                    resp = gia.generate_text(model=model, prompt=prompt, max_output_tokens=1024)
                    # common fields
                    if hasattr(resp, "text") and resp.text:
                        return str(resp.text)[:max_chars]
                    candidates = getattr(resp, "candidates", None) or getattr(resp, "outputs", None)
                    if candidates:
                        try:
                            if isinstance(candidates, list) and len(candidates) > 0:
                                first = candidates[0]
                                if isinstance(first, dict):
                                    for key in ("content", "text", "output", "message"):
                                        val = first.get(key)
                                        if isinstance(val, str) and val:
                                            return val[:max_chars]
                                        if isinstance(val, list) and len(val) > 0:
                                            for item in val:
                                                if isinstance(item, str):
                                                    return item[:max_chars]
                                                if isinstance(item, dict):
                                                    for vv in item.values():
                                                        if isinstance(vv, str):
                                                            return vv[:max_chars]
                        except Exception:
                            pass

                if hasattr(gia, "responses"):
                    resp = gia.responses.create(model=model, input=prompt)
                    if hasattr(resp, "output_text") and resp.output_text:
                        return str(resp.output_text)[:max_chars]
                    if hasattr(resp, "candidates") and resp.candidates:
                        try:
                            c = resp.candidates[0]
                            txt = getattr(c, "output", None) or getattr(c, "content", None)
                            if isinstance(txt, list) and len(txt) > 0:
                                return str(txt[0].get("text") or txt[0].get("content") or "")[:max_chars]
                        except Exception:
                            pass

                # last resort generic extract and debug
                got = generic_extract_text(resp)
                if got:
                    return got
                logging.error("google.generativeai client returned an unexpected response structure. Dumping response for debugging.")
                debug_dump_response(resp)
            except Exception as e:
                logging.error("google.generativeai client call failed: %s", e)

        logging.error("Could not interpret any Gemini/GenAI client response format.")
        return None
    except Exception as e:
        logging.error("Gemini summarization failed: %s", e)
        return None


def local_summarize(text: str, max_chars: int) -> str:
    """Basic local summarizer fallback:
    - Strip HTML
    - Collapse whitespace
    - Take as many full sentences as fit within max_chars, otherwise truncate cleanly
    This is a safe, dependency-free fallback used only when Gemini fails.
    """
    try:
        # Strip basic HTML tags
        import re

        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean).strip()

        # Split into sentences (naive)
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        out = []
        for s in sentences:
            if not s:
                continue
            candidate = (" ".join(out) + " " + s).strip() if out else s
            if len(candidate) <= max_chars:
                out.append(s)
            else:
                # if nothing added yet, truncate the sentence
                if not out:
                    return s[:max_chars].rstrip()
                break

        result = " ".join(out).strip()
        if not result:
            # fallback to truncation
            result = clean[:max_chars].rstrip()
        return result
    except Exception as e:
        logging.error("Local summarizer failed: %s", e)
        return text[:max_chars]


def build_caption(title: str, summary: str) -> str:
    # keep this function minimal; tags will be inserted by caller when available
    caption = f"<b>{title}</b>\n\n{summary}"
    return caption


def make_tags_line(tag_names: List[str]) -> str:
    if tag_names and len(tag_names) > 0:
        # sanitize and make hashtags
        import re

        hashtags = []
        for t in tag_names[:2]:
            cleaned = re.sub(r"[^A-Za-z0-9]", "", t)
            if cleaned:
                hashtags.append(f"#{cleaned}")
        if hashtags:
            return " ".join(hashtags)

    # default
    return "#new #job"


def send_with_retry(func, *args, max_retries: int = 3, initial_delay: float = 1.0, **kwargs):
    """Helper to retry a send function (telebot methods) with exponential backoff."""
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning("Attempt %d failed: %s", attempt, e)
            if attempt == max_retries:
                logging.error("All %d attempts failed.", max_retries)
                raise
            time.sleep(delay)
            delay *= 2


def send_text_safely(bot: telebot.TeleBot, channel_id: str, text: str, reply_markup=None):
    """Split long text into Telegram-friendly chunks (<=4096 chars) and send sequentially."""
    MAX_LEN = 4000
    parts = []
    if len(text) <= MAX_LEN:
        parts = [text]
    else:
        # naive split on paragraph boundaries
        paragraphs = text.split('\n\n')
        cur = ""
        for p in paragraphs:
            chunk = (cur + "\n\n" + p).strip() if cur else p
            if len(chunk) <= MAX_LEN:
                cur = chunk
            else:
                if cur:
                    parts.append(cur)
                # if paragraph itself is too long, split by sentence
                import re
                sentences = re.split(r'(?<=[.!?])\s+', p)
                cur2 = ""
                for s in sentences:
                    cand = (cur2 + " " + s).strip() if cur2 else s
                    if len(cand) <= MAX_LEN:
                        cur2 = cand
                    else:
                        parts.append(cand[:MAX_LEN])
                        cur2 = ""
                if cur2:
                    cur = cur2
        if cur:
            parts.append(cur)

    last_msg = None
    for i, part in enumerate(parts):
        try:
            if i == len(parts) - 1 and reply_markup:
                last_msg = send_with_retry(bot.send_message, channel_id, part, parse_mode="HTML", reply_markup=reply_markup)
            else:
                last_msg = send_with_retry(bot.send_message, channel_id, part, parse_mode="HTML")
        except Exception as e:
            logging.error("Failed to send text part: %s", e)
            raise
    return last_msg


def send_photo_with_caption_safely(bot: telebot.TeleBot, channel_id: str, photo_url: str, caption: str, reply_markup=None):
    """Send photo and ensure caption respects 1024-char limit; if caption too long, send remainder as separate message."""
    try:
        if len(caption) <= 1024:
            return send_with_retry(bot.send_photo, channel_id, photo_url, caption=caption, parse_mode="HTML", reply_markup=reply_markup)
        else:
            head = caption[:1024]
            tail = caption[1024:]
            send_with_retry(bot.send_photo, channel_id, photo_url, caption=head, parse_mode="HTML")
            return send_text_safely(bot, channel_id, tail, reply_markup=reply_markup)
    except Exception as e:
        logging.error("Failed to send photo safely: %s", e)
        raise


def escape_markdown(text: str) -> str:
    # Basic escape for Markdown special chars; Telegram's Markdown is forgiving but this reduces issues.
    replace_chars = "_*[]()~`>#+-=|{}.!"
    for ch in replace_chars:
        text = text.replace(ch, f"\\{ch}")
    return text


def post_to_telegram(bot: telebot.TeleBot, channel_id: str, post: dict, summary: str):
    title = post["title"]
    link = post["link"]
    featured = post.get("featured_image")
    # Build tags
    tag_names = extract_tags_from_post(post)
    tags_line = make_tags_line(tag_names)

    # Contact sentence (corrected English) where users can contact this bot to add new vacancies
    contact_sentence = "📢@UrjiiJobsVacancy \n 🤖@UrjiiJob_bot"

    # Build caption: Title, Detail (summary), Tags, mention, contact sentence
    caption_body = build_caption(title, summary)
    full_caption = f"{caption_body}\n\n{tags_line}\n\n{contact_sentence}"

    # Keyboard with Apply Here
    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton(text="Apply Here", url=link))

    # Send safely handling caption limits and text chunking
    try:
        if featured:
            send_photo_with_caption_safely(bot, channel_id, featured, full_caption, reply_markup=keyboard)
        else:
            send_text_safely(bot, channel_id, full_caption, reply_markup=keyboard)
    except Exception as e:
        logging.error("Failed to post to Telegram for post %s: %s", post.get("id"), e)


def process_new_posts(bot: telebot.TeleBot, channel_id: str, posts: List[dict], posted_ids: List[int], gemini_api_key: Optional[str]):
    updated = False
    for raw in posts:
        p = extract_post_fields(raw)
        pid = p["id"]
        if pid in posted_ids:
            continue

        logging.info("Processing new post %s: %s", pid, p["title"])

        # Determine target length
        max_chars = 800 if p.get("featured_image") else 1500

        # Summarize
        summary = None
        try:
            if gemini_api_key:
                summary = summarize_with_gemini(p["content"], max_chars)
            else:
                logging.warning("GEMINI_API_KEY not provided; using local summarizer fallback.")
        except Exception as e:
            logging.error("Summarization error for post %s: %s", pid, e)

        # If Gemini failed or returned None, use local fallback
        if not summary:
            logging.info("Using local summarizer fallback for post %s", pid)
            summary = local_summarize(p["content"], max_chars)

        # Trim and basic clean
        summary = summary.strip()

        # Post to Telegram
        try:
            post_to_telegram(bot, channel_id, p, summary)
            posted_ids.append(pid)
            updated = True
        except Exception as e:
            logging.error("Failed to post job %s: %s", pid, e)

    if updated:
        save_posted_ids(posted_ids)


def run_loop(use_sample: bool = False):
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHANNEL_ID = os.environ.get("TELEGRAM_CHANNEL_ID")
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID:
        logging.error("Environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID must be set.")
        return

    if GEMINI_API_KEY:
        configure_gemini(GEMINI_API_KEY)

    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

    posted_ids = load_posted_ids()

    def job():
        logging.info("Fetching posts...")
        posts = fetch_posts(use_sample=use_sample)
        if not isinstance(posts, list):
            logging.error("Unexpected posts format: expected list, got %s", type(posts))
            return
        process_new_posts(bot, TELEGRAM_CHANNEL_ID, posts, posted_ids, GEMINI_API_KEY)

    # Run once at start
    job()

    # Schedule every 5 minutes
    schedule.every(2).minutes.do(job)

    logging.info("Started polling every 5 minutes. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post new WP jobs to Telegram with Gemini summaries.")
    parser.add_argument("--use-sample", action="store_true", help="Use local res.json as sample input instead of live API.")
    args = parser.parse_args()
    run_loop(use_sample=args.use_sample)

#!/usr/bin/env python3
"""
whatsapp_monitor.py - WhatsApp Message Monitor with LID mapping support.
When a client is added, they receive a welcome message listing all available MCP tools.
"""

import json
import logging
import os
import queue
import signal
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import sys
import re
import requests

sys.path.insert(0, '../')
from streamlit_mcp_app import SyncMCPClient, MultiLLMEngine, ConversationMemory

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SERVERS_CONFIG_PATH = os.environ.get("MCP_CONFIG", "../servers.yaml")
if not os.path.exists(SERVERS_CONFIG_PATH):
    alt_path = os.path.join(os.path.dirname(__file__), SERVERS_CONFIG_PATH)
    if os.path.exists(alt_path):
        SERVERS_CONFIG_PATH = alt_path
    else:
        raise FileNotFoundError(
            f"MCP config not found at '{SERVERS_CONFIG_PATH}'. "
            f"Set MCP_CONFIG env var or ensure servers.yaml exists."
        )
STATE_FILE = Path("whatsapp_monitor_state.json")
DEFAULT_ADMINS = {"94722314386"}

# Default LID mappings to ensure at startup (e.g., admin's own LID)
DEFAULT_LID_MAP = {
    "276252413931735": "94722314386"
}

POLL_INTERVAL = 10 # Seconds between DB polls for new messages
WORKER_THREADS = 4 # Number of worker threads to process messages concurrently

# Conversation memory settings (used only when USE_CONTEXT = True)
MAX_CONVERSATION_MESSAGES = 2          # Max messages to include in LLM context
CONVERSATION_MAX_AGE_MINUTES = 6       # Only include messages from last N minutes
# Set to False to disable any conversation history (stateless mode)
USE_CONTEXT = True

# ----- New configuration flags -----
ALLOW_GROUPS = True               # Set to True to include group chats (@g.us)
ALLOW_STATUS_BROADCAST = False     # Set to True to include status@broadcast (not recommended)
# -----------------------------------

# ----- Auto LID mapping on /add -----
AUTO_MAP_LID_ON_ADD = True
LID_LOOKUP_TIMEOUT = 10 # Seconds to wait for JID to appear in DB after adding client before giving up on auto-mapping
LID_LOOKUP_POLL_INTERVAL = 1 # Seconds between DB polls when waiting for JID after /add command
BRIDGE_API_URL = "http://localhost:8080/api"   # Go bridge API endpoint

# ----- Rate Limiting -----
RATE_LIMIT_CHAT_PER_SEC = 1.0
RATE_LIMIT_CHAT_BURST = 3
RATE_LIMIT_GLOBAL_PER_SEC = 0.25
RATE_LIMIT_GLOBAL_BURST = 15

class RateLimiter:
    def __init__(self, tokens_per_sec: float, max_burst: int):
        self.per_sec = tokens_per_sec
        self.max_burst = max_burst
        self.tokens: Dict[str, float] = {}
        self.last_refill: Dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        if key not in self.tokens:
            self.tokens[key] = self.max_burst
            self.last_refill[key] = now
        elapsed = now - self.last_refill.get(key, now)
        self.tokens[key] = min(self.max_burst, self.tokens[key] + elapsed * self.per_sec)
        self.last_refill[key] = now
        if self.tokens[key] >= 1.0:
            self.tokens[key] -= 1.0
            return True
        return False

# ----- Scheduling Configuration -----
SCHEDULER_ENABLED = True                     # Set to False to disable auto news
SCHEDULER_DEFAULT_INTERVAL_SECONDS = 3600    # 1 hour fallback
SCHEDULER_NEWS_LIMIT = 15                    # Number of headlines to send
SCHEDULER_NEWS_TYPE = "headlines"            # "headlines" or "breaking"

# ----- Direct news request bypass (no LLM) -----
NEWS_KEYWORDS = {
    "news", "headlines", "headline", "breaking", "latest",
    "sri lanka news", "today's news", "what's happening",
    "update me", "give me news", "show me news"
}

# ----- LLM Model selection -----
LLM_MODEL = os.environ.get("WHATSAPP_LLM_MODEL", "gpt-5-nano")   # can be "deepseek-chat", "gpt-5-nano", etc.

class LogCaptureHandler(logging.Handler):
    def __init__(self, capacity: int = 100):
        super().__init__()
        from collections import deque
        self.buffer = deque(maxlen=capacity)
    def emit(self, record: logging.LogRecord):
        self.buffer.append(self.format(record))
    def get_contents(self) -> str:
        return "\n".join(list(self.buffer))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("whatsapp_monitor")
log_capture = LogCaptureHandler()
logger.addHandler(log_capture)
# logger.setLevel(logging.DEBUG)   # uncomment to enable DEBUG
# -----------------------------------------------------------------------------
# Timestamp Conversion
# -----------------------------------------------------------------------------

def iso_to_db_format(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    except ValueError:
        dt = datetime.strptime(iso_str[:19], "%Y-%m-%dT%H:%M:%S")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
    offset = dt.strftime("%z")
    offset_formatted = offset[:3] + ":" + offset[3:] if len(offset) == 5 else offset
    return dt.strftime("%Y-%m-%d %H:%M:%S") + offset_formatted

def db_to_iso_format(db_ts: str) -> str:
    return db_ts.replace(' ', 'T')

# -----------------------------------------------------------------------------
# Helper: News request detection
# -----------------------------------------------------------------------------
def is_news_request(text: str) -> bool:
    text_lower = text.lower().strip()
    if text_lower.startswith("/news"):
        return True
    for kw in NEWS_KEYWORDS:
        if kw in text_lower:
            return True
    return False

# -----------------------------------------------------------------------------
# Persistent State with LID Mapping and Blocked List (extended with scheduler settings)
# -----------------------------------------------------------------------------

class MonitorState:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load state from file, merging default admins, clients, LID mappings, and scheduler settings if missing."""
        default_data = {
            "last_processed_timestamp": None,
            "admins": list(DEFAULT_ADMINS),
            "clients": [],
            "lid_map": {},
            "blocked": [],
            # Scheduler settings
            "scheduled_phones": [],
            "scheduler_interval": SCHEDULER_DEFAULT_INTERVAL_SECONDS,
            "scheduler_enabled": SCHEDULER_ENABLED,
            "scheduler_news_limit": SCHEDULER_NEWS_LIMIT,
            "scheduler_news_type": SCHEDULER_NEWS_TYPE,
        }

        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                # Ensure all expected keys exist
                for key, default_value in default_data.items():
                    if key not in data:
                        data[key] = default_value
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                data = default_data
        else:
            data = default_data

        # Merge default admins (add any that are missing)
        for admin in DEFAULT_ADMINS:
            if admin not in data["admins"]:
                data["admins"].append(admin)
                logger.info(f"Added default admin: {admin}")

        # Merge default LID mappings (do not overwrite existing)
        for lid, phone in DEFAULT_LID_MAP.items():
            if lid not in data["lid_map"]:
                data["lid_map"][lid] = phone
                logger.info(f"Added default LID mapping: {lid} → {phone}")

        return data

    def save(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    @property
    def last_processed_timestamp(self) -> Optional[str]:
        return self.data.get("last_processed_timestamp")

    @last_processed_timestamp.setter
    def last_processed_timestamp(self, value: Optional[str]):
        self.data["last_processed_timestamp"] = value
        self.save()

    @property
    def last_processed_cursor(self) -> tuple:
        return (
            self.data.get("last_message_id"),
            self.data.get("last_chat_jid"),
        )

    @last_processed_cursor.setter
    def last_processed_cursor(self, value: tuple):
        msg_id, chat_jid = value
        self.data["last_message_id"] = msg_id
        self.data["last_chat_jid"] = chat_jid
        self.save()

    @property
    def admins(self) -> Set[str]:
        return set(self.data.get("admins", []))

    def add_admin(self, phone: str) -> bool:
        if phone not in self.admins:
            self.data["admins"].append(phone)
            self.save()
            return True
        return False

    def remove_admin(self, phone: str) -> bool:
        if phone in self.admins:
            self.data["admins"].remove(phone)
            self.save()
            return True
        return False

    @property
    def clients(self) -> Set[str]:
        return set(self.data.get("clients", []))

    def add_client(self, phone: str) -> bool:
        if phone not in self.clients:
            self.data["clients"].append(phone)
            # Remove from blocked if present
            if phone in self.blocked:
                self.data["blocked"].remove(phone)
            self.save()
            return True
        return False

    def remove_client(self, phone: str) -> bool:
        if phone in self.clients:
            self.data["clients"].remove(phone)
            # Also remove any LID mappings for this phone
            lids_to_remove = [lid for lid, mapped_phone in self.data["lid_map"].items() if mapped_phone == phone]
            for lid in lids_to_remove:
                del self.data["lid_map"][lid]
                logger.info(f"Removed LID mapping {lid} → {phone} during client removal")
            self.save()
            return True
        return False

    @property
    def blocked(self) -> Set[str]:
        return set(self.data.get("blocked", []))

    def add_blocked(self, phone: str) -> bool:
        if phone not in self.blocked:
            self.data["blocked"].append(phone)
            # Remove from clients and admins if present
            if phone in self.clients:
                self.data["clients"].remove(phone)
            if phone in self.admins:
                self.data["admins"].remove(phone)
            self.save()
            return True
        return False

    def remove_blocked(self, phone: str) -> bool:
        if phone in self.blocked:
            self.data["blocked"].remove(phone)
            self.save()
            return True
        return False

    def map_lid(self, lid: str, phone: str) -> bool:
        self.data["lid_map"][lid] = phone
        self.save()
        return True

    def unmap_lid(self, lid: str) -> bool:
        if lid in self.data["lid_map"]:
            del self.data["lid_map"][lid]
            self.save()
            return True
        return False

    def get_mapped_phone(self, lid: str) -> Optional[str]:
        return self.data.get("lid_map", {}).get(lid)

    # Scheduler methods
    @property
    def scheduled_phones(self) -> List[str]:
        return self.data.get("scheduled_phones", [])

    def add_scheduled_phone(self, phone: str) -> bool:
        if phone not in self.scheduled_phones:
            self.data["scheduled_phones"].append(phone)
            self.save()
            return True
        return False

    def remove_scheduled_phone(self, phone: str) -> bool:
        if phone in self.scheduled_phones:
            self.data["scheduled_phones"].remove(phone)
            self.save()
            return True
        return False

    @property
    def scheduler_interval(self) -> int:
        return self.data.get("scheduler_interval", SCHEDULER_DEFAULT_INTERVAL_SECONDS)

    @scheduler_interval.setter
    def scheduler_interval(self, value: int):
        self.data["scheduler_interval"] = value
        self.save()

    @property
    def scheduler_enabled(self) -> bool:
        return self.data.get("scheduler_enabled", SCHEDULER_ENABLED)

    @scheduler_enabled.setter
    def scheduler_enabled(self, value: bool):
        self.data["scheduler_enabled"] = value
        self.save()

    @property
    def scheduler_news_limit(self) -> int:
        return self.data.get("scheduler_news_limit", SCHEDULER_NEWS_LIMIT)

    @scheduler_news_limit.setter
    def scheduler_news_limit(self, value: int):
        self.data["scheduler_news_limit"] = value
        self.save()

    @property
    def scheduler_news_type(self) -> str:
        return self.data.get("scheduler_news_type", SCHEDULER_NEWS_TYPE)

    @scheduler_news_type.setter
    def scheduler_news_type(self, value: str):
        if value in ("headlines", "breaking"):
            self.data["scheduler_news_type"] = value
            self.save()

# -----------------------------------------------------------------------------
# JID Cache (fixed resolve_canonical_phone to use lid_map for bare user strings)
# -----------------------------------------------------------------------------

class JIDCache:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.environ.get("WHATSAPP_DB_PATH")
            if not db_path:
                candidates = [
                    os.path.join(os.path.dirname(__file__), "..", "whatsapp-bridge", "store", "messages.db"),
                    "/home/erangadesaram/Documents/Eranga/Dev/MCP-Udemy/MCP_Server_with_WhatsApp/whatsapp-mcp/whatsapp-bridge/store/messages.db",
                ]
                db_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        self.db_path = db_path
        self.cache: Dict[str, str] = {}
        self.lock = threading.Lock()

    def get_jid(self, phone_number: str) -> Optional[str]:
        with self.lock:
            if phone_number in self.cache:
                return self.cache[phone_number]
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT jid FROM chats WHERE jid LIKE ? AND jid NOT LIKE '%@g.us' LIMIT 1",
                (f"%{phone_number}%",),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                jid = row[0]
                with self.lock:
                    self.cache[phone_number] = jid
                return jid
        except Exception as e:
            logger.error(f"JID lookup failed for {phone_number}: {e}")
        return None

    def clear_cache(self, phone: str = None):
        """Clear cache for a specific phone or all if None."""
        with self.lock:
            if phone:
                if phone in self.cache:
                    del self.cache[phone]
                jids_to_remove = [jid for jid, p in self.cache.items() if p == phone]
                for jid in jids_to_remove:
                    del self.cache[jid]
            else:
                self.cache.clear()

    def resolve_canonical_phone(self, jid: str, lid_map: Dict[str, str] = None) -> str:
        """
        Convert a sender identifier (full JID or user part) to a canonical phone number.
        Uses internal cache, lid_map, and database lookup.
        """
        with self.lock:
            if jid in self.cache:
                return self.cache[jid]

        # Case 1: No '@' – likely just the user part (phone or LID)
        if "@" not in jid:
            # Check if it's a mapped LID (numeric string)
            if lid_map and jid in lid_map:
                phone = lid_map[jid]
                with self.lock:
                    self.cache[jid] = phone
                return phone
            # Otherwise, assume it's already a phone number
            return jid

        # Case 2: Full JID with '@'
        local, server = jid.split("@", 1)

        # LID with mapping
        if server == "lid" and lid_map and local in lid_map:
            phone = lid_map[local]
            with self.lock:
                self.cache[jid] = phone
            return phone

        # Standard WhatsApp user JID
        if server == "s.whatsapp.net":
            with self.lock:
                self.cache[jid] = local
            return local

        # For any other server (e.g., groups), try to get the name from chats table
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM chats WHERE jid = ?", (jid,))
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                name = row[0].strip()
                if name.isdigit():
                    with self.lock:
                        self.cache[jid] = name
                    return name
                else:
                    with self.lock:
                        self.cache[jid] = jid
                    return jid
        except Exception as e:
            logger.error(f"Failed to resolve canonical phone for {jid}: {e}")

        return jid

# -----------------------------------------------------------------------------
# Monitor Core (extended with Scheduler and News Bypass)
# -----------------------------------------------------------------------------

class WhatsAppMonitor:
    def __init__(
        self,
        start_from_timestamp: Optional[str] = None,
        only_unread: bool = False,
        debug_allow_all: bool = False,
    ):
        self.start_from_timestamp = start_from_timestamp
        self.only_unread = only_unread
        self.debug_allow_all = debug_allow_all

        self.state = MonitorState(STATE_FILE)
        self.jid_cache = JIDCache()

        if start_from_timestamp is not None:
            self.last_processed_iso = start_from_timestamp
        elif self.state.last_processed_timestamp:
            self.last_processed_iso = self.state.last_processed_timestamp
        else:
            latest_db = self._get_latest_db_timestamp()
            if latest_db:
                self.last_processed_iso = db_to_iso_format(latest_db)
            else:
                self.last_processed_iso = (datetime.now() - timedelta(hours=1)).isoformat()

        self.last_processed_db = iso_to_db_format(self.last_processed_iso)
        logger.info(f"Using last_processed (DB format): {self.last_processed_db}")

        self.mcp_client = SyncMCPClient()
        logger.info("Connecting to MCP servers...")
        result = self.mcp_client.connect(SERVERS_CONFIG_PATH, timeout=400)
        if result["successful"] == 0:
            raise RuntimeError("No MCP servers connected.")
        logger.info(f"Connected to {result['successful']} server(s).")

        self.engine = MultiLLMEngine(self.mcp_client, model=LLM_MODEL)
        logger.info(f"Using LLM model: {self.engine.model}")

        self.message_queue = queue.Queue(maxsize=1000)

        self.running = False
        self.poll_thread = None
        self.worker_threads = []
        # Scheduler attributes
        self.scheduler_thread = None
        self.scheduler_last_run = 0

        self.chat_rate_limiter = RateLimiter(RATE_LIMIT_CHAT_PER_SEC, RATE_LIMIT_CHAT_BURST)
        self.global_rate_limiter = RateLimiter(RATE_LIMIT_GLOBAL_PER_SEC, RATE_LIMIT_GLOBAL_BURST)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _get_latest_db_timestamp(self) -> Optional[str]:
        try:
            conn = sqlite3.connect(self.jid_cache.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp FROM messages ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception:
            return None

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received.")
        self.stop()

    def start(self):
        if self.running:
            return
        self.running = True

        self.poll_thread = threading.Thread(target=self._poll_loop, name="PollThread", daemon=True)
        self.poll_thread.start()

        for i in range(WORKER_THREADS):
            t = threading.Thread(target=self._worker_loop, name=f"Worker-{i}", daemon=True)
            t.start()
            self.worker_threads.append(t)

        # Start scheduler thread if enabled
        if self.state.scheduler_enabled:
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, name="SchedulerThread", daemon=True)
            self.scheduler_thread.start()
            logger.info(f"Scheduler started (interval={self.state.scheduler_interval}s)")
        else:
            logger.info("Scheduler is disabled in state.")

        logger.info("Monitor started. Press Ctrl+C to stop.")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        if not self.running:
            return
        logger.info("Shutting down...")
        self.running = False

        if self.poll_thread and self.poll_thread.is_alive():
            self.poll_thread.join(timeout=5)

        for t in self.worker_threads:
            if t.is_alive():
                t.join(timeout=5)

        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)

        self.state.last_processed_timestamp = self.last_processed_iso
        self.state.save()
        self.mcp_client.cleanup(timeout=10)
        logger.info("Monitor stopped.")

    # -------------------------------------------------------------------------
    # Scheduler Loop
    # -------------------------------------------------------------------------
    def _scheduler_loop(self):
        while self.running:
            if not self.state.scheduler_enabled:
                time.sleep(5)
                continue

            now = time.time()
            interval = self.state.scheduler_interval
            if now - self.scheduler_last_run >= interval:
                self._run_scheduled_news()
                self.scheduler_last_run = now
            time.sleep(10)

    def _run_scheduled_news(self):
        phones = self.state.scheduled_phones
        if not phones:
            logger.info("Scheduler: No phone numbers in scheduled_phones list.")
            return

        logger.info(f"Scheduler: Fetching news (type={self.state.scheduler_news_type}, limit={self.state.scheduler_news_limit})")
        try:
            if self.state.scheduler_news_type == "breaking":
                news_text = self.mcp_client.execute_tool(
                    "sri_lanka_news",
                    "get_breaking_news",
                    {"limit": self.state.scheduler_news_limit},
                    timeout=30
                )
            else:
                news_text = self.mcp_client.execute_tool(
                    "sri_lanka_news",
                    "get_headlines",
                    {"limit": self.state.scheduler_news_limit},
                    timeout=30
                )
        except Exception as e:
            logger.error(f"Scheduler failed to fetch news: {e}")
            return

        if not news_text or len(news_text.strip()) < 10:
            logger.warning("Scheduler: Retrieved empty or very short news text.")
            return

        header = f"📰 *Sri Lanka News Digest*\n_{datetime.now().strftime('%Y-%m-%d %H:%M')}_\n\n"
        full_message = header + news_text
        if len(full_message) > 60000:
            full_message = full_message[:60000] + "\n\n... (truncated)"

        for phone in phones:
            phone = phone.strip()
            if not phone:
                continue
            try:
                result = self.mcp_client.execute_tool(
                    "whatsapp_automation",
                    "send_message",
                    {"recipient": phone, "message": full_message},
                    timeout=60
                )
                if "error" not in result.lower() and "failed" not in result.lower():
                    logger.info(f"Scheduler: Sent news to {phone}")
                else:
                    logger.error(f"Scheduler: Failed to send to {phone}: {result}")
            except Exception as e:
                logger.exception(f"Scheduler: Exception sending to {phone}: {e}")

    # -------------------------------------------------------------------------
    # Poll & Worker Methods
    # -------------------------------------------------------------------------
    def _poll_loop(self):
        while self.running:
            try:
                new_messages = self._fetch_new_messages()
                for msg in new_messages:
                    if msg.get("is_from_me", False):
                        continue
                    self.message_queue.put(msg, timeout=1)
                if new_messages:
                    last_msg = new_messages[-1]
                    self.last_processed_iso = db_to_iso_format(last_msg["timestamp"])
                    self.last_processed_db = last_msg["timestamp"]
                    self.state.last_processed_timestamp = self.last_processed_iso
                    self.state.last_processed_cursor = (
                        last_msg["id"],
                        last_msg["chat_jid"],
                    )
            except Exception as e:
                logger.error(f"Poll error: {e}")
            time.sleep(POLL_INTERVAL)

    def _fetch_new_messages(self) -> List[Dict]:
        db_path = self.jid_cache.db_path
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            last_msg_id, last_chat_jid = self.state.last_processed_cursor
            if last_msg_id and last_chat_jid:
                where_clauses = [
                    "(messages.timestamp > ? OR (messages.timestamp = ? AND messages.id > ?))"
                ]
                params = [self.last_processed_db, self.last_processed_db, last_msg_id]
            else:
                where_clauses = ["messages.timestamp > ?"]
                params = [self.last_processed_db]

            if not ALLOW_STATUS_BROADCAST:
                where_clauses.append("chats.jid NOT LIKE '%@broadcast'")
            if not ALLOW_GROUPS:
                where_clauses.append("chats.jid NOT LIKE '%@g.us'")

            where_sql = " AND ".join(where_clauses)

            query = f"""
                SELECT 
                    messages.timestamp,
                    messages.sender,
                    chats.name as chat_name,
                    messages.content,
                    messages.is_from_me,
                    messages.chat_jid,
                    messages.id,
                    messages.media_type
                FROM messages
                JOIN chats ON messages.chat_jid = chats.jid
                WHERE {where_sql}
                ORDER BY messages.timestamp ASC, messages.id ASC
            """
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            messages = []
            for row in rows:
                msg = dict(row)
                msg["timestamp"] = row["timestamp"]
                messages.append(msg)

            if messages:
                logger.info(f"Fetched {len(messages)} new messages.")
            return messages

        except Exception as e:
            logger.exception(f"DB query failed: {e}")
            return []

    def _worker_loop(self):
        while self.running:
            try:
                msg = self.message_queue.get(timeout=1)
                self._process_message(msg)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Worker error: {e}")

    # -------------------------------------------------------------------------
    # Direct news fetch (bypass LLM)
    # -------------------------------------------------------------------------
    def _fetch_and_format_news(self) -> str:
        try:
            news_text = self.mcp_client.execute_tool(
                "sri_lanka_news",
                "get_headlines",
                {"limit": 20},
                timeout=30
            )
            if not news_text or len(news_text.strip()) < 10:
                return "No news available at the moment."
            if len(news_text) > 60000:
                news_text = news_text[:60000] + "\n\n... (truncated)"
            return f"📰 *Latest Sri Lanka News*\n\n{news_text}"
        except Exception as e:
            logger.error(f"Direct news fetch error: {e}")
            return ""

    # -------------------------------------------------------------------------
    # Message processing (with news bypass)
    # -------------------------------------------------------------------------
    def _process_message(self, msg: Dict):
        chat_jid = msg["chat_jid"]
        sender_jid = msg["sender"]
        content = msg["content"].strip()
        timestamp = msg["timestamp"]

        if not ALLOW_STATUS_BROADCAST and "@broadcast" in chat_jid:
            logger.info(f"Ignored message from status broadcast: {chat_jid}")
            return
        if not ALLOW_GROUPS and "@g.us" in chat_jid:
            logger.info(f"Ignored message from group (disabled): {chat_jid}")
            return

        sender_identifier = self.jid_cache.resolve_canonical_phone(
            sender_jid,
            lid_map=self.state.data.get("lid_map", {})
        )

        # Check blocked list first (highest priority)
        if sender_identifier in self.state.blocked:
            logger.info(f"Ignored – {sender_identifier} is blocked")
            return

        logger.info(f"Message from {sender_identifier} (JID: {sender_jid}): {content[:50]}...")

        is_admin = sender_identifier in self.state.admins
        is_client = sender_identifier in self.state.clients

        if self.debug_allow_all:
            is_admin = True

        if is_admin and content.startswith("/"):
            reply = self._handle_admin_command(sender_identifier, content, chat_jid)
            if reply:
                self._send_reply(chat_jid, reply)
            return

        if not (is_admin or is_client):
            logger.info(f"Ignored – {sender_identifier} is not admin/client")
            return

        # ----- Direct news bypass (no LLM) -----
        if is_news_request(content):
            logger.info(f"Direct news request from {sender_identifier}")
            news_reply = self._fetch_and_format_news()
            self._send_reply(chat_jid, news_reply if news_reply else "Sorry, news service unavailable.")
            return

        # Normal LLM processing
        if USE_CONTEXT:
            memory = self._build_contextual_memory(chat_jid, timestamp)
        else:
            memory = self._build_stateless_memory()

        memory.add_message("user", content)

        # Retry up to 3 times on transient errors (timeouts, etc.)
        max_attempts = 3
        answer = ""
        for attempt in range(max_attempts):
            try:
                answer, updated_conv, _ = self.engine.process_conversation(
                    memory.get_messages(), manual_limit=2
                )
                if answer.strip():
                    break   # got a valid response
                else:
                    logger.warning(f"LLM returned empty response (attempt {attempt+1}/{max_attempts}).")
                    if attempt < max_attempts - 1:
                        time.sleep(1)
            except Exception as e:
                is_transient = "Timeout" in str(e) or "timed out" in str(e).lower() or "Connection" in str(e)
                if is_transient and attempt < max_attempts - 1:
                    logger.warning(f"Transient error (attempt {attempt+1}/{max_attempts}): {e}")
                    time.sleep(2)
                else:
                    logger.exception("LLM error")
                    answer = f"Error: {e}"
                    break
        
        if answer.strip():
            self._send_reply(chat_jid, answer)
        else:
            logger.warning("LLM returned empty response after all retries.")
            self._send_reply(
                chat_jid,
                "I'm sorry, I couldn't generate a response right now. Please try again in a moment."
            )    

    def _build_contextual_memory(self, chat_jid: str, current_timestamp: str) -> ConversationMemory:
        mem = ConversationMemory(max_messages=MAX_CONVERSATION_MESSAGES)
        prompt_path = Path(__file__).parent / "system_prompt.txt"
        system_prompt = prompt_path.read_text() if prompt_path.exists() else "You are a helpful assistant with access to MCP tools."
        mem.add_message("system", system_prompt)

        try:
            current_dt = datetime.fromisoformat(db_to_iso_format(current_timestamp))
            cutoff_dt = current_dt - timedelta(minutes=CONVERSATION_MAX_AGE_MINUTES)
            cutoff_db = iso_to_db_format(cutoff_dt.isoformat())

            conn = sqlite3.connect(self.jid_cache.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = """
                SELECT timestamp, sender, content, is_from_me
                FROM messages
                WHERE chat_jid = ? AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp ASC
                LIMIT ?
            """
            cursor.execute(query, (chat_jid, cutoff_db, current_timestamp, MAX_CONVERSATION_MESSAGES))
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                role = "assistant" if row["is_from_me"] else "user"
                mem.add_message(role, row["content"])

            logger.debug(f"Loaded {len(rows)} recent messages for context.")
        except Exception as e:
            logger.error(f"Failed to build conversation memory: {e}")

        return mem

    def _build_stateless_memory(self) -> ConversationMemory:
        mem = ConversationMemory(max_messages=1)
        mem.add_message(
            "system",
            "You are a helpful assistant with access to MCP tools. "
            "Respond concisely and accurately to the user's query. "
            "Ignore any non-existent conversation history."
        )
        return mem

    def _get_available_tools_excluding_whatsapp(self) -> str:
        try:
            servers = self.mcp_client.get_servers(timeout=10)
        except Exception as e:
            logger.error(f"Failed to get servers list: {e}")
            return "• Various MCP tools are available (could not retrieve detailed list)."

        lines = []
        for srv_name, srv_info in servers.items():
            if srv_name.lower() in ("whatsapp", "whatsapp_automation"):
                continue
            tools = srv_info.get("tools", [])
            if tools:
                lines.append(f"\n🔧 **{srv_name}**")
                for tool in tools[:10]:
                    desc = tool.get("description", "No description")[:80]
                    lines.append(f"  • `{tool['name']}` – {desc}")
        if not lines:
            return "• No additional MCP tools are currently available."
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Bridge API helper
    # -------------------------------------------------------------------------
    def _resolve_jid_via_api(self, phone: str) -> Optional[str]:
        try:
            url = f"{BRIDGE_API_URL}/resolve_jid"
            response = requests.get(url, params={"phone": phone}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data.get("jid")
            logger.debug(f"Bridge API did not return JID for {phone}: {response.text}")
        except Exception as e:
            logger.warning(f"Failed to call bridge API for JID resolution: {e}")
        return None

    # -------------------------------------------------------------------------
    # Improved /add with LID resolution
    # -------------------------------------------------------------------------
    def _add_client_with_lid_resolution(self, phone: str) -> tuple[bool, str]:
        logger.info(f"Starting intelligent /add for {phone}")
        self.state.add_client(phone)

        processing_msg = (
            "🔄 Hi, I'm an automated assistant,\n"
            "*You are being added as a client!*\n"
            "Request is processing please wait..."
        )
        sent_ok = self._send_message_to_phone(phone, processing_msg)
        if not sent_ok:
            logger.warning(f"Failed to send processing message to {phone}")
            return True, f"✅ Added {phone}. ⚠️ Could not send initial message (number may not be on WhatsApp)."

        jid = self._resolve_jid_via_api(phone)
        if jid:
            logger.info(f"Bridge API resolved JID for {phone}: {jid}")
        else:
            logger.info(f"Bridge API failed, falling back to DB polling for {phone}")
            jid = self._wait_for_jid(phone, timeout=LID_LOOKUP_TIMEOUT, poll_interval=LID_LOOKUP_POLL_INTERVAL)

        if jid:
            if "@lid" in jid:
                lid = jid.split("@")[0]
                self.state.map_lid(lid, phone)
                logger.info(f"Auto-mapped LID {lid} → {phone}")
            elif "@s.whatsapp.net" in jid:
                # Standard JID – mapping not required, phone already known
                pass

            welcome_msg = self._get_welcome_message()
            self._send_reply(jid, welcome_msg)
            return True, f"✅ Added {phone} and sent full welcome message (JID {jid})."
        else:
            return True, (
                f"✅ Added {phone}. ⚠️ JID not found yet. "
                f"The client will be authorized once they send their first message. "
                f"You can manually map their LID later with `/maplid <lid> {phone}`."
            )

    def _wait_for_jid(self, phone: str, timeout: int, poll_interval: float) -> Optional[str]:
        start = time.time()
        while time.time() - start < timeout:
            jid = self.jid_cache.get_jid(phone)
            if jid:
                return jid
            time.sleep(poll_interval)
        return None

    def _add_client_fallback(self, phone: str) -> tuple[bool, str]:
        self.state.add_client(phone)
        client_jid = self.jid_cache.get_jid(phone)
        if client_jid:
            welcome_msg = self._get_welcome_message()
            self._send_reply(client_jid, welcome_msg)
            return True, f"✅ Added {phone} and sent welcome message (JID {client_jid})."
        else:
            return True, f"✅ Added {phone}. JID not found – welcome will be sent on first contact."

    def _send_message_to_phone(self, phone: str, text: str) -> bool:
        try:
            result = self.mcp_client.execute_tool(
                "whatsapp_automation",
                "send_message",
                {"recipient": phone, "message": text},
                timeout=30
            )
            try:
                data = json.loads(result)
                success = data.get("success", False)
            except json.JSONDecodeError:
                success = "error" not in result.lower() and "failed" not in result.lower()
            if success:
                logger.info(f"✅ Processing message sent to {phone}")
            else:
                logger.error(f"❌ Send to {phone} failed: {result}")
            return success
        except Exception as e:
            logger.exception(f"💥 Exception sending to {phone}: {e}")
            return False

    def _get_welcome_message(self) -> str:
        tools_list = self._get_available_tools_excluding_whatsapp()
        return f"""
*👋 Hello Dear Client! You have been added to the WhatsApp Bot!*.

This is an automated message from the MCP Client. I'm reaching out to inform you about the capabilities of our MCP client system.

The MCP (Model Context Protocol) client is connected to multiple servers with various tools available:

*Available Tools from MCP Servers:*

1.*🔧Remote Test Server Web Search Tools:*
    -Web search functionality with multiple search engines
    -File operations (read/write/list directories)
    -Cryptocurrency price lookup
    -Topic analysis and historical report generation
    -Inventory management tools
    -Note-taking and memory storage
    -Person database management

2.*🌐Open Web Search Tools:*
    -Advanced web search with Brave, DuckDuckGo, and Exa engines
    -Content fetching from various platforms (GitHub, CSDN, Linux.do, Juejin)
    -Web content extraction with readability options

3.*🧠 Personal Knowledge Graph Memory Store:*
   - **Entity Management**: Create, read, update, and delete knowledge entities
   - **Relationship Mapping**: Establish and manage connections between concepts
   - **Observations Storage**: Store detailed observations and insights
   - **Graph Search**: Intelligent search across knowledge networks
   - **Memory Persistence**: Long-term storage of structured knowledge
   - **Contextual Retrieval**: Access relevant information based on semantic relationships

4.*♟️CGit Chess Statistics Tools:*
   -Chess.com player profile retrieval
   -Chess player statistics analysis

*Key Capabilities:*
- **Knowledge Management**: Structured storage and retrieval of information using entity-relationship modeling
- **Multi-engine web searching**: Comprehensive information gathering across multiple search platforms
- **File management and content extraction**: Document processing and data extraction capabilities
- **Data analysis and report generation**: Automated analysis and professional reporting
- **Contact and database management**: Structured data organization and retrieval
- **Cross-platform content fetching**: Aggregation of information from diverse online sources
- **Intelligent memory systems**: Context-aware knowledge storage and retrieval

*Professional Applications:*
- **Research & Development**: Organize research findings, track project insights, and maintain knowledge continuity
- **Client Relationship Management**: Store client preferences, interaction history, and personalized insights
- **Project Documentation**: Maintain structured project knowledge, lessons learned, and team insights
- **Learning & Development**: Track learning progress, store educational insights, and build knowledge networks
- **Decision Support**: Access historical context and related information for informed decision-making

This system enables comprehensive information gathering, analysis, and automation across various domains.

Best regards, 
MCP Client System                                                
"""

    # -------------------------------------------------------------------------
    # Admin Command Handler (with enhanced removal and block commands + model switch + scheduler & news bypass commands)
    # -------------------------------------------------------------------------
    def _handle_admin_command(self, sender_phone: str, command: str, chat_jid: str) -> Optional[str]:
        parts = command.split()
        if not parts:
            return None
        cmd = parts[0].lower()

        if cmd == "/add" and len(parts) >= 2:
            phone = re.sub(r'\s+', '', ' '.join(parts[1:]))
            if phone in self.state.clients:
                return f"ℹ️ {phone} is already a client."
            if AUTO_MAP_LID_ON_ADD:
                success, message = self._add_client_with_lid_resolution(phone)
                return message
            else:
                success, message = self._add_client_fallback(phone)
                return message

        if cmd == "/remove" and len(parts) >= 2:
            phone = re.sub(r'\s+', '', ' '.join(parts[1:]))
            if self.state.remove_client(phone):
                self.jid_cache.clear_cache(phone)
                return f"✅ Removed {phone} (LID mappings also cleared)."
            return f"ℹ️ {phone} not a client."

        if cmd == "/block" and len(parts) >= 2:
            phone = re.sub(r'\s+', '', ' '.join(parts[1:]))
            if self.state.add_blocked(phone):
                self.jid_cache.clear_cache(phone)
                return f"🚫 Blocked {phone}. They will be ignored."
            return f"ℹ️ {phone} is already blocked."

        if cmd == "/unblock" and len(parts) >= 2:
            phone = re.sub(r'\s+', '', ' '.join(parts[1:]))
            if self.state.remove_blocked(phone):
                return f"✅ Unblocked {phone}."
            return f"ℹ️ {phone} is not blocked."

        if cmd == "/list":
            clients = sorted(self.state.clients)
            return "📋 Clients:\n" + "\n".join(clients) if clients else "No clients."

        if cmd == "/listblocked":
            blocked = sorted(self.state.blocked)
            return "🚫 Blocked:\n" + "\n".join(blocked) if blocked else "No blocked numbers."

        if cmd == "/admin":
            if len(parts) < 2:
                return "Usage: /admin [add|remove|list] [phone]"
            subcmd = parts[1].lower()
            if subcmd == "add" and len(parts) >= 3:
                phone = re.sub(r'\s+', '', ' '.join(parts[2:]))
                if self.state.add_admin(phone):
                    return f"✅ Added {phone} as admin."
                return f"ℹ️ {phone} is already an admin."
            elif subcmd == "remove" and len(parts) >= 3:
                phone = re.sub(r'\s+', '', ' '.join(parts[2:]))
                if len(self.state.admins) <= 1:
                    return "❌ Cannot remove the last admin."
                if self.state.remove_admin(phone):
                    return f"✅ Removed admin {phone}."
                return f"ℹ️ {phone} is not an admin."
            elif subcmd == "list":
                admins = sorted(self.state.admins)
                return "👑 Admins:\n" + "\n".join(admins)
            else:
                return "Usage: /admin add <phone>, /admin remove <phone>, /admin list"

        if cmd == "/maplid" and len(parts) >= 3:
            lid = re.sub(r'\s+', '', ' '.join(parts[1:2]))
            phone = re.sub(r'\s+', '', ' '.join(parts[2:]))
            if self.state.map_lid(lid, phone):
                return f"✅ Mapped LID {lid} → {phone}."
            return "❌ Failed to map LID."

        if cmd == "/unmaplid" and len(parts) >= 2:
            lid = re.sub(r'\s+', '', ' '.join(parts[1:]))
            if self.state.unmap_lid(lid):
                return f"✅ Removed mapping for LID {lid}."
            return f"ℹ️ No mapping found for LID {lid}."

        if cmd == "/listlids":
            mappings = self.state.data.get("lid_map", {})
            if mappings:
                lines = [f"{lid} → {phone}" for lid, phone in mappings.items()]
                return "📋 LID Mappings:\n" + "\n".join(lines)
            return "📋 No LID mappings."

        if cmd == "/model":
            if len(parts) >= 2:
                new_model = parts[1].lower()
                allowed = {"deepseek-chat", "gpt-5-nano"}
                if new_model not in allowed:
                    return f"❌ Unsupported model. Allowed: {', '.join(sorted(allowed))}."
                old_model = self.engine.model
                self.engine = MultiLLMEngine(self.mcp_client, model=new_model)
                logger.info(f"Admin switched LLM from {old_model} to {new_model}")
                return f"✅ Switched LLM model from {old_model} to {new_model}."
            else:
                return f"Current LLM model: {self.engine.model}"

        # ----- News bypass test command -----
        if cmd == "/testnews":
            news = self._fetch_and_format_news()
            return news if news else "No news fetched."

        # ----- Scheduler commands -----
        if cmd == "/schedulerenable":
            self.state.scheduler_enabled = True
            if not self.scheduler_thread or not self.scheduler_thread.is_alive():
                self.scheduler_thread = threading.Thread(target=self._scheduler_loop, name="SchedulerThread", daemon=True)
                self.scheduler_thread.start()
            return "✅ Scheduled news delivery ENABLED."

        if cmd == "/schedulerdisable":
            self.state.scheduler_enabled = False
            return "⏸️ Scheduled news delivery DISABLED."

        if cmd == "/setinterval" and len(parts) >= 2:
            try:
                interval = int(parts[1])
                if interval < 60:
                    return "❌ Minimum interval is 60 seconds."
                self.state.scheduler_interval = interval
                return f"✅ Scheduler interval set to {interval} seconds."
            except ValueError:
                return "❌ Invalid number. Usage: /setinterval <seconds>"

        if cmd == "/addscheduledphone" and len(parts) >= 2:
            phone = re.sub(r'\s+', '', ' '.join(parts[1:]))
            if self.state.add_scheduled_phone(phone):
                return f"✅ Added {phone} to scheduled news list."
            return f"ℹ️ {phone} already in scheduled list."

        if cmd == "/removescheduledphone" and len(parts) >= 2:
            phone = re.sub(r'\s+', '', ' '.join(parts[1:]))
            if self.state.remove_scheduled_phone(phone):
                return f"✅ Removed {phone} from scheduled news list."
            return f"ℹ️ {phone} not in scheduled list."

        if cmd == "/listscheduledphones":
            phones = self.state.scheduled_phones
            if phones:
                return "📋 Scheduled phones:\n" + "\n".join(phones)
            return "📋 No scheduled phones."

        if cmd == "/setnewstype" and len(parts) >= 2:
            typ = parts[1].lower()
            if typ not in ("headlines", "breaking"):
                return "❌ Type must be 'headlines' or 'breaking'."
            self.state.scheduler_news_type = typ
            return f"✅ Scheduled news type set to {typ}."

        if cmd == "/setnewslimit" and len(parts) >= 2:
            try:
                limit = int(parts[1])
                if limit < 1 or limit > 50:
                    return "❌ Limit must be between 1 and 50."
                self.state.scheduler_news_limit = limit
                return f"✅ Scheduled news limit set to {limit}."
            except ValueError:
                return "❌ Invalid number."

        if cmd == "/schedulerstatus":
            status = "🟢 ENABLED" if self.state.scheduler_enabled else "🔴 DISABLED"
            return (f"Scheduler: {status}\n"
                    f"Interval: {self.state.scheduler_interval}s\n"
                    f"News type: {self.state.scheduler_news_type}\n"
                    f"Limit: {self.state.scheduler_news_limit}\n"
                    f"Scheduled phones: {', '.join(self.state.scheduled_phones) or 'none'}")

        if cmd == "/logs":
            return "📋 Recent logs:\n" + log_capture.get_contents()[-3000:]

        if cmd == "/help":
            return (
                "/add <phone> - Add a client\n"
                "/remove <phone> - Remove a client (clears LID mapping)\n"
                "/block <phone> - Block a number completely\n"
                "/unblock <phone> - Remove block\n"
                "/list - List clients\n"
                "/listblocked - List blocked numbers\n"
                "/admin add <phone> - Add an admin\n"
                "/admin remove <phone> - Remove an admin\n"
                "/admin list - List admins\n"
                "/maplid <lid> <phone> - Map a LID to a real number\n"
                "/unmaplid <lid> - Remove a LID mapping\n"
                "/listlids - Show all LID mappings\n"
                "/model <model_name> - Switch LLM model (deepseek-chat, gpt-5-nano)\n"
                "/testnews - Get news directly (bypass LLM)\n"
                "/schedulerenable - Enable auto news\n"
                "/schedulerdisable - Disable auto news\n"
                "/setinterval <seconds> - Set interval (min 60s)\n"
                "/addscheduledphone <phone> - Add to auto-news\n"
                "/removescheduledphone <phone> - Remove from auto-news\n"
                "/listscheduledphones - Show auto-news list\n"
                "/setnewstype <headlines|breaking>\n"
                "/setnewslimit <num>\n"
                "/schedulerstatus - Show settings\n"
                "/logs - Show recent log entries\n"
                "/help - Show this help"
            )

        return f"Unknown command '{cmd}'. Try /help."

    def _send_reply(self, chat_jid: str, text: str):
        if "@broadcast" in chat_jid:
            logger.warning(f"Refusing to send reply to broadcast JID: {chat_jid}")
            return

        if not self.chat_rate_limiter.allow(chat_jid):
            logger.warning(f"Chat rate limit hit for {chat_jid}, dropping reply")
            return
        if not self.global_rate_limiter.allow("global"):
            logger.warning("Global rate limit hit, dropping reply")
            return

        logger.info(f"Sending reply to {chat_jid}: {text[:100]}...")
        try:
            result = self.mcp_client.execute_tool(
                "whatsapp_automation",
                "send_message",
                {"recipient": chat_jid, "message": text},
                timeout=60
            )
            try:
                data = json.loads(result)
                success = data.get("success", False)
                msg = data.get("message", "")
            except json.JSONDecodeError:
                success = "error" not in result.lower() and "failed" not in result.lower()
                msg = result
            if success:
                logger.info(f"✅ Reply sent to {chat_jid}")
            else:
                logger.error(f"❌ Send failed: {msg}")
        except Exception as e:
            logger.exception(f"💥 Send exception: {e}")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=str) # ISO timestamp to start from (overrides saved state)
    parser.add_argument("--only-unread", action="store_true") # For future use: currently not implemented, but could be used to filter messages in _fetch_new_messages()
    parser.add_argument("--debug-allow-all", action="store_true") # For testing: allow all messages regardless of admin/client status
    args = parser.parse_args()

    monitor = WhatsAppMonitor(
        start_from_timestamp=args.start_from,
        only_unread=args.only_unread,
        debug_allow_all=args.debug_allow_all,
    )
    monitor.start()

if __name__ == "__main__":
    main()
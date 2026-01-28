# milvus_diag.py
import os
import sys
import socket
from typing import Optional, Tuple, List
from dotenv import load_dotenv

print("🔧 Loading .env ...")
load_dotenv()  # loads from current working dir

def ping(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def try_connect(desc: str, **kwargs) -> Tuple[bool, str]:
    """
    kwargs will be passed to pymilvus.connections.connect(...)
    Returns (ok, message)
    """
    try:
        # Import here to avoid import noise
        from pymilvus import connections, utility
        alias = kwargs.pop("alias", "diag")
        # if alias exists, remove it to re-connect cleanly
        try:
            connections.remove_connection(alias)
        except Exception:
            pass
        connections.connect(alias=alias, **kwargs)
        cols = utility.list_collections(using=alias)
        msg = f"✅ {desc} CONNECTED. Collections: {cols}"
        # cleanup
        try:
            connections.remove_connection(alias)
        except Exception:
            pass
        return True, msg
    except Exception as e:
        return False, f"❌ {desc} FAILED: {repr(e)}"

def env(k: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(k)
    return v if (v is not None and str(v).strip() != "") else default

# Read env
URI   = env("MILVUS_URI")
TOKEN = env("MILVUS_TOKEN")
HOST  = env("MILVUS_HOST")
PORT  = env("MILVUS_PORT", "19530")
USER  = env("MILVUS_USER", "")
PASS  = env("MILVUS_PASSWORD", "")
DB    = env("MILVUS_DB_NAME", "default")
SEC   = env("MILVUS_SECURE", "false")

print("\n🌍 Current env values:")
print(f"  MILVUS_URI   = {URI}")
print(f"  MILVUS_TOKEN = {'***' if TOKEN else None}")
print(f"  MILVUS_HOST  = {HOST}")
print(f"  MILVUS_PORT  = {PORT}")
print(f"  MILVUS_USER  = {USER}")
print(f"  MILVUS_DB    = {DB}")
print(f"  MILVUS_SECURE= {SEC}")

print("\n🔎 Trying connection permutations...\n")

results: List[str] = []

# 1) URI + TOKEN
if URI and TOKEN:
    ok, msg = try_connect("URI+TOKEN",
                          uri=URI, token=TOKEN, db_name=DB, alias="u_token")
    results.append(msg)
    print(msg)

# 2) URI + USER/PASS
if URI and (USER or PASS):
    ok, msg = try_connect("URI+USER/PASS",
                          uri=URI, user=USER or None, password=PASS or None, db_name=DB, alias="u_user")
    results.append(msg)
    print(msg)

# 3) HOST:PORT with secure False / True, try both port 19530 and 19531
host_ports = []
if HOST:
    # preferred env port first
    try:
        env_port_int = int(PORT)
        host_ports.append(env_port_int)
    except Exception:
        pass
    # add common ports ensuring no duplicates
    for p in (19530, 19531):
        if p not in host_ports:
            host_ports.append(p)

    for p in host_ports:
        # quick TCP check to save time
        reachable = ping(HOST, p, timeout=1.0)
        print(f"  • TCP test {HOST}:{p} -> {'OK' if reachable else 'NO REACH'}")

        for secure_flag in (False, True):
            ok, msg = try_connect(f"HOST:{HOST}:{p} secure={secure_flag}",
                                  host=HOST, port=str(p), user=USER or None,
                                  password=PASS or None, db_name=DB, secure=secure_flag, alias=f"h_{p}_{int(secure_flag)}")
            results.append(msg)
            print(msg)

# If nothing was attempted due to missing envs
if not results:
    print("⚠️ No attempts made. Please set one of the following in .env:")
    print("   • MILVUS_URI + MILVUS_TOKEN   (Zilliz Cloud)")
    print("   • OR MILVUS_HOST + MILVUS_PORT (+ MILVUS_SECURE=true if TLS)")

# Summary
print("\n====== SUMMARY ======")
for r in results:
    print(r)

# exit code: 0 if any success
sys.exit(0 if any(r.startswith("✅") for r in results) else 1)

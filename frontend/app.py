"""
Flask frontend for the Custom Research AI Agent.

Requirements:
    pip install flask requests ddgs python-dotenv --break-system-packages

Setup:
    Place your .env file (with OPENROUTER_API_KEY) in this same folder,
    OR keep ai_agent.py + .env one level up — see the import path below.

Run:
    python app.py
    Then open http://127.0.0.1:5000 in your browser.
"""

import os
import sys
import time
import uuid
import threading

from flask import Flask, request, jsonify, render_template

# --- import the agent from ai_agent.py -----------------------------------
# This assumes ai_agent.py sits in the parent folder of this frontend/ dir.
# If you instead copy ai_agent.py into this same folder, change this to:
#     from ai_agent import run_agent
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ai_agent import run_agent  # noqa: E402

app = Flask(__name__)

# In-memory job store: {job_id: {"status", "log": [...], "report": str}}
JOBS = {}


def run_job(job_id: str, topic: str):
    JOBS[job_id]["status"] = "running"

    def on_search(query):
        JOBS[job_id]["log"].append(query)

    try:
        report = run_agent(topic, on_search=on_search)
        JOBS[job_id]["report"] = report
        JOBS[job_id]["status"] = "done"
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/research", methods=["POST"])
def start_research():
    data = request.get_json(force=True)
    topic = (data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Topic is required"}), 400

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "queued", "log": [], "report": None, "topic": topic}

    thread = threading.Thread(target=run_job, args=(job_id, topic), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/research/<job_id>")
def poll_research(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify(job)


if __name__ == "__main__":
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "⚠️  OPENROUTER_API_KEY not found in environment.\n"
            "   Make sure your .env file is set up correctly.\n"
        )
    # threaded=True lets the browser poll for job status while the
    # background research thread is still running.
    # use_reloader=False stops Flask from restarting the process mid-job,
    # which was killing the background thread and dropping the connection.
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from datetime import datetime, date
from functools import wraps

# ─── Phase 2: AI Anomaly Detection imports ───────────────────────────────────
import numpy as np
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
app.secret_key = os.urandom(24)

DB_PATH = "fitguard.db"

# ─── Database Setup ─────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    NOT NULL UNIQUE,
                email    TEXT    NOT NULL UNIQUE,
                password TEXT    NOT NULL,
                created_at TEXT  DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS workouts (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER NOT NULL,
                workout_type   TEXT    NOT NULL,
                duration       INTEGER NOT NULL,
                calories       INTEGER,
                difficulty     TEXT,
                notes          TEXT,
                workout_date   TEXT    NOT NULL,
                created_at     TEXT    DEFAULT CURRENT_TIMESTAMP,
                anomaly_status TEXT    DEFAULT 'normal',
                anomaly_reason TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS challenge_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    challenge_id INTEGER NOT NULL,
    user_id      INTEGER NOT NULL,
    log_date     TEXT    NOT NULL,
    UNIQUE (challenge_id, log_date),          -- the actual enforcement
    FOREIGN KEY (challenge_id) REFERENCES challenges(id),
    FOREIGN KEY (user_id)      REFERENCES users(id)
);               

            CREATE TABLE IF NOT EXISTS challenges (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL,
                title       TEXT    NOT NULL,
                target_days INTEGER NOT NULL,
                current_day INTEGER DEFAULT 0,
                start_date  TEXT    NOT NULL,
                status      TEXT    DEFAULT 'active',
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
                           
        """)
        # ── Migration: add columns to existing databases that predate Phase 3 ──
        existing = {row[1] for row in conn.execute("PRAGMA table_info(workouts)")}
        if "anomaly_status" not in existing:
            conn.execute("ALTER TABLE workouts ADD COLUMN anomaly_status TEXT DEFAULT 'normal'")
        if "anomaly_reason" not in existing:
            conn.execute("ALTER TABLE workouts ADD COLUMN anomaly_reason TEXT")

init_db()

# ─── Phase 2: Train Isolation Forest on startup ──────────────────────────────
#
# We train on synthetic "normal" workout data so the model has a baseline.
# Each row is [duration_minutes, calories_burned].
#
# Normal ranges we teach the model:
#   duration : 10 – 120 minutes  (realistic everyday workouts)
#   calories : 50 – 800 kcal     (light walk → intense HIIT)
#
# The model learns this normal zone and will flag anything far outside it.

np.random.seed(42)
_normal_duration = np.random.randint(10, 121, 500)          # 500 synthetic sessions
# calories scale roughly 5–8 kcal/min for normal workouts, with ±50 kcal noise
_cal_per_min     = np.random.uniform(5.0, 8.0, 500)
_normal_calories = (_normal_duration * _cal_per_min + np.random.randint(-50, 50, 500)).clip(30, 900)
_training_data   = np.column_stack([_normal_duration, _normal_calories])

anomaly_model = IsolationForest(
    n_estimators=100,   # number of trees — 100 is a solid default
    contamination=0.05, # we expect ~5% of real logs to be suspicious
    random_state=42
)
anomaly_model.fit(_training_data)
# ─────────────────────────────────────────────────────────────────────────────

# ─── Phase 3: classify_workout() — called automatically on every log ─────────
def classify_workout(duration, calories, user_id, workout_type, workout_date):
    """
    Returns (status, reason) where status is 'normal' or 'suspicious'.
    Runs three checks in order:
      1. Rule-based: catches physiologically impossible values instantly
      2. Duplicate detection: flags identical logs on the same day
      3. Isolation Forest: statistical outlier check on duration + calories
    """
    cal = calories or 0  # treat None as 0 for math

    # ── Check 1: Rule-based hard limits ──────────────────────────────────────
    if duration > 600:
        return ("suspicious",
                f"Duration of {duration} min exceeds 10 hours — unrealistically long.")

    if cal > 0 and (cal / duration) > 30:
        rate = cal / duration
        return ("suspicious",
                f"Calorie rate of {rate:.1f} kcal/min is physiologically impossible "
                f"(human max is ~30 kcal/min).")

    # ── Check 2: Duplicate detection — same user, type, duration, calories, date
    with get_db() as conn:
        dupes = conn.execute("""
            SELECT COUNT(*) FROM workouts
            WHERE user_id = ? AND workout_type = ? AND duration = ?
              AND COALESCE(calories, 0) = ? AND workout_date = ?
        """, (user_id, workout_type, duration, cal, workout_date)).fetchone()[0]

    if dupes >= 2:
        return ("suspicious",
                f"This exact workout ({workout_type}, {duration} min, {cal} kcal) "
                f"has been logged {dupes} time(s) on {workout_date} — possible duplicate.")

    # ── Check 3: Isolation Forest statistical outlier ─────────────────────────
    sample     = np.array([[duration, cal]])
    prediction = anomaly_model.predict(sample)[0]   # -1 = outlier, +1 = normal
    score      = anomaly_model.decision_function(sample)[0]

    if prediction == -1:
        return ("suspicious",
                f"Unusual duration/calorie combination ({duration} min, {cal} kcal) "
                f"flagged as a statistical outlier (score: {score:.3f}).")

    return ("normal", f"Workout looks healthy (score: {score:.3f}).")
# ─────────────────────────────────────────────────────────────────────────────

# ─── Auth Decorator ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}! 💪", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "error")

    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        if not all([username, email, password, confirm]):
            flash("All fields are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        else:
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                        (username, email, generate_password_hash(password))
                    )
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Username or email already exists.", "error")

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You've been logged out. Keep grinding! 🔥", "info")
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    with get_db() as conn:
        # Recent workouts
        workouts = conn.execute("""
            SELECT * FROM workouts
            WHERE user_id = ?
            ORDER BY workout_date DESC, created_at DESC
            LIMIT 5
        """, (user_id,)).fetchall()

        # Stats
        stats = conn.execute("""
            SELECT
                COUNT(*)               AS total_workouts,
                COALESCE(SUM(duration), 0)  AS total_minutes,
                COALESCE(SUM(calories), 0)  AS total_calories
            FROM workouts WHERE user_id = ?
        """, (user_id,)).fetchone()

        # Streak: count consecutive days up to today
        all_dates = conn.execute("""
            SELECT DISTINCT workout_date FROM workouts
            WHERE user_id = ? ORDER BY workout_date DESC
        """, (user_id,)).fetchall()

        streak = 0
        check_date = date.today()
        date_set = {row["workout_date"] for row in all_dates}
        while str(check_date) in date_set:
            streak += 1
            check_date = date(check_date.year, check_date.month, check_date.day)
            from datetime import timedelta
            check_date -= timedelta(days=1)

        # Challenges
        challenges = conn.execute("""
            SELECT * FROM challenges WHERE user_id = ? AND status = 'active'
        """, (user_id,)).fetchall()

        # Weekly chart data (last 7 days)
        weekly = conn.execute("""
            SELECT workout_date, SUM(duration) as minutes
            FROM workouts
            WHERE user_id = ?
              AND workout_date >= date('now', '-6 days')
            GROUP BY workout_date
            ORDER BY workout_date
        """, (user_id,)).fetchall()

    return render_template("dashboard.html",
        workouts=workouts,
        stats=stats,
        streak=streak,
        challenges=challenges,
        weekly=weekly
    )

@app.route("/log", methods=["GET", "POST"])
@login_required
def log_workout():
    if request.method == "POST":
        user_id      = session["user_id"]
        workout_type = request.form.get("workout_type", "").strip()
        duration     = request.form.get("duration", "").strip()
        calories     = request.form.get("calories", "").strip() or None
        difficulty   = request.form.get("difficulty", "").strip()
        notes        = request.form.get("notes", "").strip()
        workout_date = request.form.get("workout_date") or str(date.today())

        if not workout_type or not duration:
            flash("Workout type and duration are required.", "error")
        else:
            try:
                duration = int(duration)
                calories = int(calories) if calories else None

                # ── Anomaly check BEFORE saving ───────────────────────────────
                anomaly_status, anomaly_reason = classify_workout(
                    duration, calories, user_id, workout_type, workout_date
                )

                # ── BLOCK suspicious workouts — do not insert ─────────────────
                if anomaly_status == "suspicious":
                    flash(f"⚠️ Suspicious activity blocked: {anomaly_reason}", "warning")
                    return render_template("log_workout.html",
                        today=str(date.today()),
                        prefill=request.form
                    )

                with get_db() as conn:
                    conn.execute("""
                        INSERT INTO workouts
                            (user_id, workout_type, duration, calories, difficulty,
                             notes, workout_date, anomaly_status, anomaly_reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, workout_type, duration, calories, difficulty,
                          notes, workout_date, anomaly_status, anomaly_reason))

                flash("Workout logged! You're crushing it 🏆", "success")
                return redirect(url_for("dashboard"))
            except ValueError:
                flash("Duration and calories must be numbers.", "error")

    return render_template("log_workout.html", today=str(date.today()), prefill=request.form)
@app.route("/workouts")
@login_required
def workouts():
    user_id = session["user_id"]
    with get_db() as conn:
        all_workouts = conn.execute("""
            SELECT * FROM workouts WHERE user_id = ?
            ORDER BY workout_date DESC, created_at DESC
        """, (user_id,)).fetchall()
    return render_template("workouts.html", workouts=all_workouts)

@app.route("/workout/delete/<int:workout_id>", methods=["POST"])
@login_required
def delete_workout(workout_id):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM workouts WHERE id = ? AND user_id = ?",
            (workout_id, session["user_id"])
        )
    flash("Workout deleted.", "info")
    return redirect(url_for("workouts"))

@app.route("/challenges", methods=["GET", "POST"])
@login_required
def challenges():
    user_id = session["user_id"]
    if request.method == "POST":
        title       = request.form.get("title", "").strip()
        target_days = request.form.get("target_days", "").strip()
        if title and target_days:
            with get_db() as conn:
                conn.execute("""
                    INSERT INTO challenges (user_id, title, target_days, start_date)
                    VALUES (?, ?, ?, ?)
                """, (user_id, title, int(target_days), str(date.today())))
            flash(f"Challenge '{title}' started! 🚀", "success")

    with get_db() as conn:
        all_challenges = conn.execute(
            "SELECT * FROM challenges WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall()
        today = str(date.today())

    done_today = {
    row["challenge_id"]
    for row in conn.execute(
    """
    SELECT challenge_id
    FROM challenge_logs
    WHERE user_id=? AND log_date=?
    """,
    (user_id, today)
    ).fetchall()
    }

    return render_template(
   "challenges.html",
   challenges=all_challenges,
   done_today=done_today
)

@app.route("/challenge/complete/<int:cid>", methods=["POST"])
@login_required
def complete_day(cid):
    user_id  = session["user_id"]
    today    = str(date.today())
 
    with get_db() as conn:
        ch = conn.execute(
            "SELECT * FROM challenges WHERE id = ? AND user_id = ?",
            (cid, user_id)
        ).fetchone()
 
        if not ch:
            flash("Challenge not found.", "error")
            return redirect(url_for("challenges"))
 
        if ch["status"] == "completed":
            flash(f"'{ch['title']}' is already fully completed! 🏆", "info")
            return redirect(url_for("challenges"))
 
        # ── Guard: has the user already checked in today for this challenge? ──
        already = conn.execute("""
            SELECT 1 FROM challenge_logs
            WHERE challenge_id = ? AND log_date = ?
        """, (cid, today)).fetchone()
 
        if already:
            flash("✋ You already completed today for this challenge. Come back tomorrow!", "warning")
            return redirect(url_for("challenges"))
 
        # ── All clear: record the log and advance the counter ─────────────────
        try:
            conn.execute("""
                INSERT INTO challenge_logs (challenge_id, user_id, log_date)
                VALUES (?, ?, ?)
            """, (cid, user_id, today))
        except Exception:
            # UNIQUE constraint fired — concurrent double-submit, treat as duplicate
            flash("✋ You already completed today for this challenge. Come back tomorrow!", "warning")
            return redirect(url_for("challenges"))
 
        new_day = ch["current_day"] + 1
        status  = "completed" if new_day >= ch["target_days"] else "active"
        conn.execute(
            "UPDATE challenges SET current_day = ?, status = ? WHERE id = ?",
            (new_day, status, cid)
        )
 
        if status == "completed":
            flash(f"🎉 Challenge '{ch['title']}' COMPLETED!", "success")
        else:
            flash(f"✅ Day {new_day}/{ch['target_days']} checked off! See you tomorrow.", "success")
 
    return redirect(url_for("challenges"))
 
@app.route("/challenge/delete/<int:cid>", methods=["POST"])
@login_required
def delete_challenge(cid):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM challenges WHERE id = ? AND user_id = ?",
            (cid, session["user_id"])
        )
    flash("Challenge removed.", "info")
    return redirect(url_for("challenges"))
 
 

@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    """
    Phase 2 — AI Anomaly Detection
    --------------------------------
    GET  : show the analysis form
    POST : receive duration + calories, run Isolation Forest, return verdict
    """
    result   = None   # will be "Normal Activity" or "Suspicious Activity Detected"
    reason   = None   # human-readable explanation shown to the user
    duration = None
    calories = None

    if request.method == "POST":
        try:
            duration = int(request.form.get("duration", 0))
            calories = int(request.form.get("calories", 0))

            # ── Rule-based pre-checks (catch obvious impossibilities first) ──
            if duration <= 0 or calories < 0:
                result = "Suspicious Activity Detected"
                reason = "Duration must be positive and calories cannot be negative."

            elif duration > 600:
                # 10 hours of continuous workout is not realistic
                result = "Suspicious Activity Detected"
                reason = f"Duration of {duration} min (>{600} min) is unrealistically long."

            elif calories > 0 and (calories / duration) > 30:
                # Burning more than 30 kcal/min is physiologically impossible
                result = "Suspicious Activity Detected"
                reason = (f"Calorie rate of {calories/duration:.1f} kcal/min is physiologically "
                          f"impossible (max ~30 kcal/min for elite athletes).")

            else:
                # ── ML check: ask the Isolation Forest ──────────────────────
                # -1 means outlier (anomaly), +1 means inlier (normal)
                sample     = np.array([[duration, calories]])
                prediction = anomaly_model.predict(sample)[0]
                score      = anomaly_model.decision_function(sample)[0]
                # score < 0  → more anomalous; score > 0 → more normal

                if prediction == -1:
                    result = "Suspicious Activity Detected"
                    reason = (f"The combination of {duration} min / {calories} kcal "
                              f"is statistically unusual (anomaly score: {score:.3f}).")
                else:
                    result = "Normal Activity"
                    reason = (f"Duration and calorie values look realistic "
                              f"(confidence score: {score:.3f}).")

        except (ValueError, TypeError):
            result = "Error"
            reason = "Please enter valid whole numbers for duration and calories."

    return render_template("analyze.html",
        result=result, reason=reason,
        duration=duration, calories=calories
    )

@app.route("/leaderboard")
@login_required
def leaderboard():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT u.id, u.username,
                COALESCE(SUM(
                    CASE
                      WHEN IFNULL(w.anomaly_status,'normal') != 'suspicious'
                      THEN w.duration
                      ELSE 0
                    END
                ),0) AS clean_min,

                COALESCE(SUM(
                    CASE
                      WHEN IFNULL(w.anomaly_status,'normal') != 'suspicious'
                      THEN COALESCE(w.calories,0)
                      ELSE 0
                    END
                ),0) AS clean_cal,

                COUNT(
                    CASE
                      WHEN IFNULL(w.anomaly_status,'normal')='suspicious'
                      THEN 1
                    END
                ) AS flagged

            FROM users u
            LEFT JOIN workouts w
            ON w.user_id = u.id

            GROUP BY u.id
        """).fetchall()

        cpts = {
            r["user_id"]: r["n"]
            for r in conn.execute(
                "SELECT user_id, COUNT(*) as n FROM challenges WHERE status='completed' GROUP BY user_id"
            ).fetchall()
        }

    board = sorted([
        {
            "rank": 0,
            "username": r["username"],

            "fit_points":
                (
                    r["clean_min"]
                    + (r["clean_cal"] * 0.05)
                    + (cpts.get(r["id"],0) * 10)
                ),

            "flagged": r["flagged"],

            "is_me": r["id"] == session["user_id"]
        }

        for r in rows

    ], key=lambda x: -x["fit_points"])[:50]

    for i, entry in enumerate(board):
        entry["rank"] = i + 1

    return render_template(
        "leaderboard.html",
        board=board
    )
@app.route("/assistant", methods=["POST"])
@login_required
def assistant():
    """
    Rule-based FitGuard Assistant — no LLM, no external calls.
    Reads the user's own workout + challenge data and returns a plain-text reply.
    """
    from flask import jsonify
    user_id = session["user_id"]
    msg     = request.json.get("message", "").strip().lower()
 
    # ── Pull user context from DB ────────────────────────────────────────────
    with get_db() as conn:
        recent = conn.execute("""
            SELECT workout_type, duration, calories, difficulty,
                   workout_date, anomaly_status, anomaly_reason
            FROM workouts WHERE user_id = ?
            ORDER BY workout_date DESC, created_at DESC LIMIT 10
        """, (user_id,)).fetchall()
 
        stats = conn.execute("""
            SELECT COUNT(*) AS total,
                   COALESCE(AVG(duration), 0)            AS avg_dur,
                   COALESCE(AVG(COALESCE(calories,0)),0) AS avg_cal,
                   COALESCE(SUM(duration), 0)            AS total_min,
                   COUNT(CASE WHEN IFNULL(anomaly_status,'normal')='suspicious'
                              THEN 1 END)                AS flagged
            FROM workouts WHERE user_id = ?
        """, (user_id,)).fetchone()
 
        challenges = conn.execute("""
            SELECT title, target_days, current_day, status
            FROM challenges WHERE user_id = ? ORDER BY id DESC LIMIT 5
        """, (user_id,)).fetchall()
 
        # Types logged in the last 14 days
        recent_types = [r["workout_type"] for r in conn.execute("""
            SELECT DISTINCT workout_type FROM workouts
            WHERE user_id = ? AND workout_date >= date('now','-14 days')
        """, (user_id,)).fetchall()]
 
        # Was yesterday a rest day?
        yesterday_count = conn.execute("""
            SELECT COUNT(*) FROM workouts
            WHERE user_id = ? AND workout_date = date('now','-1 day')
        """, (user_id,)).fetchone()[0]
 
        # Last flagged workout
        last_flag = conn.execute("""
            SELECT workout_type, duration, anomaly_reason FROM workouts
            WHERE user_id = ? AND anomaly_status = 'suspicious'
            ORDER BY workout_date DESC, created_at DESC LIMIT 1
        """, (user_id,)).fetchone()
 
    # ── Shorthand helpers ────────────────────────────────────────────────────
    total      = stats["total"] or 0
    avg_dur    = round(stats["avg_dur"] or 0)
    avg_cal    = round(stats["avg_cal"] or 0)
    total_min  = stats["total_min"] or 0
    flagged    = stats["flagged"] or 0
    last_type  = recent[0]["workout_type"] if recent else None
    last_dur   = recent[0]["duration"]     if recent else 0
    active_ch  = [c for c in challenges if c["status"] == "active"]
    done_ch    = [c for c in challenges if c["status"] == "completed"]
    no_data    = total == 0
 
    # ── Intent → reply map ───────────────────────────────────────────────────
    # Each intent checks keywords in `msg` then assembles a reply from real data.
 
    # Greeting / help
    if any(k in msg for k in ("hi", "hello", "hey", "help", "what can you")):
        reply = (
            "👋 Hey! I'm your FitGuard Assistant. I can help with:\n\n"
            "• **recommend** — workout suggestions based on your history\n"
            "• **recovery** — rest & recovery advice\n"
            "• **anomaly** — explain any flagged workouts\n"
            "• **progress** — summary of your stats\n"
            "• **challenge** — status of your active challenges\n"
            "• **streak** — your consistency\n\n"
            "Just type a keyword or ask naturally!"
        )
 
    # Progress / stats summary
    elif any(k in msg for k in ("progress", "stats", "summary", "overview", "how am i")):
        if no_data:
            reply = "📊 No workouts logged yet. Log your first session to see your stats!"
        else:
            flag_note = f" ⚠️ {flagged} flagged." if flagged else " ✅ No flags."
            reply = (
                f"📊 **Your FitGuard Summary**\n\n"
                f"• Total sessions: {total}\n"
                f"• Total time: {total_min} min\n"
                f"• Avg duration: {avg_dur} min/session\n"
                f"• Avg calories: {avg_cal} kcal/session\n"
                f"• Challenges completed: {len(done_ch)}\n"
                f"• Flagged workouts: {flagged}.{flag_note}"
            )
 
    # Recommendations
    elif any(k in msg for k in ("recommend", "suggest", "what should i", "next workout", "what workout")):
        if no_data:
            reply = (
                "🏋️ You haven't logged any workouts yet — start with a 30-min session "
                "of whatever you enjoy: running, cycling, or bodyweight training."
            )
        else:
            # Suggest a type NOT logged recently to add variety
            all_types  = ["Running","Cycling","Yoga","Weight Training","Swimming",
                          "HIIT","Bodyweight","Jump Rope","Pilates","Rowing"]
            fresh      = [t for t in all_types if t not in recent_types] or all_types
            suggestion = fresh[0]
            intensity  = "Moderate"
            if avg_dur > 60:
                intensity = "Hard"
            elif avg_dur < 25:
                intensity = "Easy"
            rest_note = " Yesterday was a rest day — good to push a bit harder today! 💪" \
                        if yesterday_count == 0 else ""
            reply = (
                f"💡 **Recommended next workout:**\n\n"
                f"• Type: {suggestion}\n"
                f"• Duration: {avg_dur} min (matches your average)\n"
                f"• Intensity: {intensity} (based on your history)\n"
                f"• Target calories: ~{avg_cal} kcal\n\n"
                f"You've been logging a lot of {', '.join(recent_types[:2]) if recent_types else 'varied workouts'} lately "
                f"— mixing in {suggestion} will add variety.{rest_note}"
            )
 
    # Recovery
    elif any(k in msg for k in ("recover", "rest", "sore", "tired", "fatigue", "sleep", "overtraining")):
        if no_data:
            reply = "😴 No workout history yet, but a good rule: rest 1–2 days per week and sleep 7–9 hrs."
        else:
            # Count workouts in last 7 days
            consecutive = sum(1 for r in recent if r["workout_date"] >= str(
                date.today().replace(day=date.today().day - 7)
            )) if len(recent) >= 7 else len(recent)
 
            if consecutive >= 6:
                advice = (
                    f"⚠️ You've had {consecutive} sessions in the last 7 days — "
                    f"that's high volume! Consider a full rest day or light yoga/stretching today."
                )
            elif yesterday_count == 0:
                advice = "✅ You rested yesterday — you should be ready for a solid session today."
            elif last_dur and last_dur > 75:
                advice = (
                    f"Your last session was {last_dur} min — fairly long. "
                    f"A lighter recovery session (20–30 min walk or yoga) would be ideal today."
                )
            else:
                advice = "You look well-recovered based on recent logs. Listen to your body!"
 
            reply = (
                f"😴 **Recovery Advice**\n\n"
                f"{advice}\n\n"
                f"**General rules:**\n"
                f"• Rest 1–2 days/week minimum\n"
                f"• Sleep 7–9 hours\n"
                f"• Hydrate: 2–3L water on training days\n"
                f"• Foam roll or stretch after intense sessions"
            )
 
    # Anomaly / flag explanation
    elif any(k in msg for k in ("anomaly", "flag", "suspicious", "flagged", "blocked", "why was", "explain")):
        if flagged == 0:
            reply = "✅ Great news — none of your workouts have been flagged as suspicious!"
        elif last_flag:
            reply = (
                f"🚩 **About your flagged workout:**\n\n"
                f"• Type: {last_flag['workout_type']}\n"
                f"• Duration: {last_flag['duration']} min\n"
                f"• Reason: {last_flag['anomaly_reason']}\n\n"
                f"**What to do:** If this was a genuine session, you can mark it as "
                f"'reviewed' from your workout history. Flagged workouts don't count toward "
                f"your FitPoints until reviewed.\n\n"
                f"**Common reasons workouts get flagged:**\n"
                f"• Duration over 600 min (10+ hours)\n"
                f"• Calorie rate above 30 kcal/min (physiologically impossible)\n"
                f"• Exact duplicate logged on the same day\n"
                f"• Statistical outlier vs. normal workout patterns"
            )
        else:
            reply = f"⚠️ You have {flagged} flagged workout(s). Visit your History page to review them."
 
    # Challenge status
    elif any(k in msg for k in ("challenge", "goal", "mission")):
        if not challenges:
            reply = "🎯 No challenges started yet. Head to the Challenges page to kick one off!"
        elif active_ch:
            ch    = active_ch[0]
            pct   = round(ch["current_day"] / ch["target_days"] * 100)
            reply = (
                f"🎯 **Active Challenge: {ch['title']}**\n\n"
                f"• Progress: Day {ch['current_day']} of {ch['target_days']} ({pct}%)\n"
                f"• Remaining: {ch['target_days'] - ch['current_day']} days\n\n"
                + (f"You also have {len(active_ch)-1} more active challenge(s).\n" if len(active_ch) > 1 else "")
                + (f"✅ Completed challenges: {len(done_ch)}" if done_ch else "Keep going! 💪")
            )
        else:
            reply = f"✅ All {len(done_ch)} challenge(s) completed! Start a new one to keep the momentum."
 
    # Streak
    elif any(k in msg for k in ("streak", "consistent", "consecutive")):
        if no_data:
            reply = "🔥 Log your first workout to start a streak!"
        else:
            all_dates = [r["workout_date"] for r in recent]
            streak    = 0
            from datetime import timedelta
            check     = date.today()
            date_set  = set(all_dates)
            while str(check) in date_set:
                streak += 1
                check  -= timedelta(days=1)
            if streak == 0:
                reply = "🔥 No active streak today — log a workout to get one going!"
            elif streak < 3:
                reply = f"🔥 You're on a {streak}-day streak! Keep showing up — 3 days is where habits form."
            elif streak < 7:
                reply = f"🔥 {streak}-day streak! You're building real consistency. Push for 7!"
            else:
                reply = f"🔥 {streak}-day streak — that's elite consistency. Protect it!"
 
    # Calories / nutrition adjacent
    elif any(k in msg for k in ("calorie", "nutrition", "diet", "eat", "food", "protein")):
        reply = (
            f"🥗 **Nutrition basics for your training:**\n\n"
            f"• Your average burn: ~{avg_cal} kcal/session\n"
            f"• Eat protein within 30–60 min post-workout (0.3–0.4g per kg bodyweight)\n"
            f"• Carbs before long sessions (>45 min) for sustained energy\n"
            f"• Hydrate: add 500ml water per 30 min of intense exercise\n\n"
            f"I don't have access to your diet — a registered dietitian can give personalised advice."
        )
 
    # Motivation
    elif any(k in msg for k in ("motivat", "inspire", "tired of", "give up", "quit", "bored")):
        quotes = [
            "\"The only bad workout is the one that didn't happen.\"",
            "\"Small daily improvements are the key to staggering long-term results.\"",
            "\"Discipline is doing it even when you don't feel like it.\"",
            "\"Your future self is watching you right now through your memories.\"",
        ]
        import hashlib
        q = quotes[int(hashlib.md5(str(date.today()).encode()).hexdigest(), 16) % len(quotes)]
        sessions_note = f"You've already logged {total} session(s) — that's real work. " if total else ""
        reply = f"💪 {sessions_note}\n\n{q}\n\nYou've got this. One session at a time."
 
    # Fallback
    else:
        reply = (
            "🤔 I didn't quite catch that. Try asking about:\n\n"
            "• **recommend** — what workout to do next\n"
            "• **recovery** — rest advice\n"
            "• **anomaly** — explain a flagged workout\n"
            "• **progress** — your stats\n"
            "• **challenge** — your active goals\n"
            "• **streak** — your consistency\n"
            "• **calories** — nutrition tips"
        )
 
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
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

    return render_template("challenges.html", challenges=all_challenges)

@app.route("/challenge/complete/<int:cid>", methods=["POST"])
@login_required
def complete_day(cid):
    with get_db() as conn:
        ch = conn.execute(
            "SELECT * FROM challenges WHERE id = ? AND user_id = ?",
            (cid, session["user_id"])
        ).fetchone()
        if ch:
            new_day = ch["current_day"] + 1
            status  = "completed" if new_day >= ch["target_days"] else "active"
            conn.execute(
                "UPDATE challenges SET current_day = ?, status = ? WHERE id = ?",
                (new_day, status, cid)
            )
            if status == "completed":
                flash(f"🎉 Challenge '{ch['title']}' COMPLETED!", "success")
            else:
                flash(f"Day {new_day}/{ch['target_days']} checked off!", "success")
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

if __name__ == "__main__":
    app.run(debug=True)
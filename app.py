from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from datetime import datetime, date
from functools import wraps

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
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                workout_type TEXT    NOT NULL,
                duration     INTEGER NOT NULL,
                calories     INTEGER,
                difficulty   TEXT,
                notes        TEXT,
                workout_date TEXT    NOT NULL,
                created_at   TEXT    DEFAULT CURRENT_TIMESTAMP,
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

init_db()

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
                with get_db() as conn:
                    conn.execute("""
                        INSERT INTO workouts
                            (user_id, workout_type, duration, calories, difficulty, notes, workout_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, workout_type, duration, calories, difficulty, notes, workout_date))
                flash("Workout logged! You're crushing it 🏆", "success")
                return redirect(url_for("dashboard"))
            except ValueError:
                flash("Duration and calories must be numbers.", "error")

    return render_template("log_workout.html", today=str(date.today()))

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

if __name__ == "__main__":
    app.run(debug=True)

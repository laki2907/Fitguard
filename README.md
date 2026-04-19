# ⚡ FitGuard — AI-Powered Fitness Challenge Tracker

A full-stack Flask web application for tracking workouts, streaks, and fitness challenges.

## Features
- 🔐 User authentication (register / login / logout)
- 📊 Dashboard with stats, streak counter, and weekly bar chart
- 🏋️ Workout logging form with type, duration, calories, difficulty & notes
- 📜 Full workout history with delete support
- 🎯 Challenge tracker with day-by-day progress
- 💾 SQLite database (zero-config)

## Project Structure
```
fitguard/
├── app.py                  # Flask app, routes, DB logic
├── fitguard.db             # SQLite DB (auto-created)
├── requirements.txt
├── static/
│   ├── css/style.css
│   └── js/app.js
└── templates/
    ├── base.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── log_workout.html
    ├── workouts.html
    └── challenges.html
```

## Setup & Run

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Usage
1. Register a new account
2. Log your first workout via **Log Workout**
3. Start a fitness challenge under **Challenges**
4. Watch your streak grow on the **Dashboard**

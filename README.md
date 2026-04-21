# ⚡ FitGuard — Gamified Fitness Tracker with Anomaly Detection & Smart Assistant

FitGuard is a full-stack Flask-based fitness platform that combines workout tracking, challenge gamification, anomaly detection, and an intelligent assistant for personalized recommendations.

## Features

### Core Fitness Tracking
- 🔐 User authentication (Register / Login / Logout)
- 🏋️ Log workouts (type, duration, calories, difficulty, notes)
- 📊 Dashboard with:
  - Streak tracking
  - Total workouts
  - Total minutes
  - Calories burned
  - Weekly activity graph

### Gamification
- 🎯 Create and manage fitness challenges
- ✅ One challenge completion allowed per day (challenge integrity protection)
- 🏆 Leaderboard using FitPoints scoring

### Anomaly Detection
- 🤖 Isolation Forest based anomaly detection
- Flags suspicious workout activity:
  - unrealistic duration
  - impossible calorie burn
  - duplicate workout logs
  - statistical outliers

### Smart Assistant
- 💬 FitGuard Assistant / Smart Coach
- Context-aware support for:
  - Workout recommendations
  - Recovery advice
  - Progress summaries
  - Challenge status
  - Anomaly explanations
  - Motivation prompts

## Tech Stack

- Python
- Flask
- SQLite
- Scikit-learn
- NumPy
- HTML/CSS
- JavaScript
- Jinja2

## Project Structure

```text
fitguard/
├── app.py
├── fitguard.db
├── requirements.txt
├── utils.py
├── static/
│   ├── css/style.css
│   └── js/
│       ├── app.js
│       └── assistant.js
└── templates/
    ├── base.html
    ├── login.html
    ├── register.html
    ├── dashboard.html
    ├── log_workout.html
    ├── workouts.html
    ├── challenges.html
    ├── leaderboard.html
    ├── analyze.html
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

For Windows:

```bash
venv\Scripts\activate
```

Run locally:

```bash
http://127.0.0.1:5000
```

## Usage

1. Create account  
2. Log workouts  
3. Start challenges  
4. Use anomaly detection  
5. Track leaderboard progress  
6. Use Smart Coach assistant for recommendations

## Future Improvements
- Adaptive challenge generation
- Burnout detection
- Expanded intelligent recommendations
- Mobile-responsive deployment

## Author
Built as a mini-project using Flask and machine learning concepts.
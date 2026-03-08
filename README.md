# NexHire AI Interviewer 🚀

An AI-powered **technical interview simulation platform** with real-time monitoring, candidate assessment, and gamified engineering challenges.

## 🔥 Features

### AI Interview Engine

* AI-generated technical interview questions
* Resume-aware question generation
* Adaptive difficulty based on candidate performance

### Real-Time Interview Room

* Secure **session PIN generation**
* **Candidate invite link generation**
* Real-time interviewer monitoring
* Live scoring and feedback

### Gamified Engineering Challenge

Includes the **Kilimanjaro Engineering Challenge**:

* Interactive question-based climbing game
* Real-time scoring and altitude tracking
* Boosters (Shield, 50/50, Time Extension)
* Candidate performance analytics

### Security Features

* Environment variable secrets
* MongoDB Atlas integration
* Session PIN protection
* Secure candidate access tokens

---

# 🧠 System Architecture

```
Frontend
   │
   ├── Interviewer Dashboard
   ├── Candidate Panel
   └── Kilimanjaro Challenge Game

Backend (Flask + SocketIO)
   │
   ├── AI Question Generation
   ├── Simulation Engine
   ├── Real-time events
   └── Game event streaming

Database
   │
   ├── MongoDB Atlas
   └── SQLite session fallback
```

---

# ⚙️ Tech Stack

**Backend**

* Python
* Flask
* Flask-SocketIO

**AI**

* Groq API
* LLaMA models
* Whisper speech-to-text

**Frontend**

* HTML
* CSS
* JavaScript

**Database**

* MongoDB Atlas
* SQLite

---

# 🚀 Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/NEXHIRE_AI_INTERVIEWER.git
cd NEXHIRE_AI_INTERVIEWER
```

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
python app.py
```

Open browser:

```
http://localhost:5000
```

---

# 🔑 Environment Variables

Create a `.env` file:

```
MONGO_URI=your_mongodb_uri
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
LIVEKIT_URL=ws://localhost:7880
GROQ_API_KEY=your_groq_key
SECRET_KEY=your_secret_key
```

---

# 🎮 Kilimanjaro Engineering Challenge

The candidate must climb **Mount Kilimanjaro (5895m)** by solving engineering problems.

Game mechanics:

* Correct answer → climb altitude
* Wrong answer → lose life
* Streaks → score multiplier
* Boosters available during gameplay

---

# 📂 Project Structure

```
NEXHIRE_AI_INTERVIEWER
│
├── app.py
├── requirements.txt
├── templates
│   ├── index.html
│   └── kilimanjaro_game.html
│
├── static
│
├── .gitignore
├── README.md
└── LICENSE
```

---

# 👨‍💻 Author

**Chandrajit**

AI Engineer & Developer

---

# ⭐ Future Improvements

* AI interviewer voice interaction
* Advanced simulation scenarios
* Multi-candidate assessment
* Cloud deployment

---

# 📜 License

This project is licensed under the MIT License.

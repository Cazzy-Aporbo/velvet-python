#!/usr/bin/env python3
"""
Advanced Educational Game Generator
Creates interactive HTML5 games from any topic using AI-like content generation
Demonstrates: OOP, Design Patterns, Metaclasses, Decorators, Context Managers, 
Async operations, Type hints, and advanced Python features
"""

import json
import random
import hashlib
import asyncio
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Protocol, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
from contextlib import contextmanager
from functools import wraps, lru_cache
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type Variables
T = TypeVar('T')
GameType = TypeVar('GameType', bound='BaseGame')

# ENUMS & CONSTANTS 
class GameDifficulty(Enum):
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()
    EXPERT = auto()

class GameMode(Enum):
    QUIZ = "quiz"
    MEMORY = "memory"
    PUZZLE = "puzzle"
    WORD_GAME = "word_game"
    MATCHING = "matching"
    TIMELINE = "timeline"

# DECORATORS 
def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        result = await func(*args, **kwargs)
        duration = (datetime.datetime.now() - start).total_seconds()
        logger.info(f"{func.__name__} took {duration:.2f} seconds")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        result = func(*args, **kwargs)
        duration = (datetime.datetime.now() - start).total_seconds()
        logger.info(f"{func.__name__} took {duration:.2f} seconds")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def validate_input(validation_func):
    """Decorator for input validation"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not validation_func(*args, **kwargs):
                raise ValueError(f"Invalid input for {func.__name__}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

# METACLASSES 
class GameMeta(type):
    """Metaclass for game registration and validation"""
    _registry: Dict[str, type] = {}
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != 'BaseGame' and not name.startswith('Abstract'):
            mcs._registry[name.lower()] = cls
            logger.info(f"Registered game type: {name}")
        return cls
    
    @classmethod
    def get_game_class(mcs, name: str) -> Optional[type]:
        return mcs._registry.get(name.lower())
    
    @classmethod
    def list_games(mcs) -> List[str]:
        return list(mcs._registry.keys())

# PROTOCOLS
class ContentGenerator(Protocol):
    """Protocol for content generation strategies"""
    def generate(self, topic: str, count: int) -> List[Dict[str, Any]]: ...

class Renderer(Protocol):
    """Protocol for rendering strategies"""
    def render(self, game_data: Dict[str, Any]) -> str: ...

#DATA CLASSES
@dataclass
class Question:
    """Represents a game question"""
    id: str = field(default_factory=lambda: hashlib.md5(str(random.random()).encode()).hexdigest()[:8])
    text: str = ""
    answers: List[str] = field(default_factory=list)
    correct_answer: str = ""
    difficulty: GameDifficulty = GameDifficulty.MEDIUM
    points: int = 10
    hint: Optional[str] = None
    explanation: Optional[str] = None
    media_url: Optional[str] = None
    
    def __post_init__(self):
        if not self.correct_answer and self.answers:
            self.correct_answer = self.answers[0]

@dataclass
class GameConfig:
    """Game configuration"""
    title: str
    topic: str
    mode: GameMode
    difficulty: GameDifficulty = GameDifficulty.MEDIUM
    question_count: int = 10
    time_limit: Optional[int] = None
    enable_hints: bool = True
    enable_sound: bool = True
    theme_color: str = "#4A90E2"
    custom_css: str = ""
    custom_js: str = ""

# CONTENT GENERATORS 
class TopicAnalyzer:
    """Analyzes topics and generates relevant content using patterns"""
    
    KNOWLEDGE_BASE = {
        "science": ["atom", "molecule", "energy", "force", "reaction", "element", "compound"],
        "history": ["event", "period", "figure", "war", "revolution", "empire", "civilization"],
        "math": ["equation", "theorem", "formula", "calculation", "geometry", "algebra", "calculus"],
        "chemistry": ["element", "reaction", "bond", "mole", "periodic table", "compound", "solution"],
        "programming": ["function", "variable", "loop", "algorithm", "data structure", "class", "method"],
        "geography": ["country", "capital", "continent", "ocean", "mountain", "river", "climate"],
    }
    
    @classmethod
    @lru_cache(maxsize=128)
    def analyze(cls, topic: str) -> Dict[str, Any]:
        """Analyze topic and extract key concepts"""
        topic_lower = topic.lower()
        related_terms = []
        category = "general"
        
        for cat, terms in cls.KNOWLEDGE_BASE.items():
            if cat in topic_lower or any(term in topic_lower for term in terms):
                category = cat
                related_terms = terms
                break
        
        return {
            "category": category,
            "related_terms": related_terms,
            "complexity": len(topic.split()),
            "key_concepts": cls._extract_concepts(topic, related_terms)
        }
    
    @staticmethod
    def _extract_concepts(topic: str, related_terms: List[str]) -> List[str]:
        """Extract key concepts from topic"""
        words = re.findall(r'\w+', topic.lower())
        concepts = [word for word in words if len(word) > 3]
        concepts.extend([term for term in related_terms if term in topic.lower()])
        return list(set(concepts))[:5]

class SmartContentGenerator:
    """Generates educational content based on topic analysis"""
    
    def __init__(self, topic: str):
        self.topic = topic
        self.analysis = TopicAnalyzer.analyze(topic)
    
    @performance_monitor
    def generate_questions(self, count: int = 10, difficulty: GameDifficulty = GameDifficulty.MEDIUM) -> List[Question]:
        """Generate questions based on topic"""
        questions = []
        templates = self._get_templates()
        
        for i in range(count):
            template = random.choice(templates)
            question = self._create_question_from_template(template, difficulty)
            questions.append(question)
        
        return questions
    
    def _get_templates(self) -> List[Dict[str, str]]:
        """Get question templates based on category"""
        category = self.analysis["category"]
        
        if category == "chemistry":
            return [
                {"q": "What is the chemical symbol for {element}?", "type": "symbol"},
                {"q": "How many moles are in {number} grams of {compound}?", "type": "calculation"},
                {"q": "What type of bond exists in {compound}?", "type": "bonding"},
            ]
        elif category == "history":
            return [
                {"q": "In what year did {event} occur?", "type": "date"},
                {"q": "Who was the leader of {country} during {period}?", "type": "figure"},
                {"q": "What was the main cause of {event}?", "type": "cause"},
            ]
        else:
            return [
                {"q": f"What is the definition of {{concept}} in {self.topic}?", "type": "definition"},
                {"q": f"Which of these is related to {self.topic}?", "type": "multiple_choice"},
                {"q": f"True or False: {{statement}} about {self.topic}", "type": "boolean"},
            ]
    
    def _create_question_from_template(self, template: Dict[str, str], difficulty: GameDifficulty) -> Question:
        """Create a question from template"""
        concepts = self.analysis["key_concepts"]
        
        # Generate question text
        question_text = template["q"]
        if "{element}" in question_text:
            question_text = question_text.replace("{element}", random.choice(["Hydrogen", "Carbon", "Oxygen", "Nitrogen"]))
        if "{compound}" in question_text:
            question_text = question_text.replace("{compound}", random.choice(["H2O", "CO2", "NaCl", "CH4"]))
        if "{concept}" in question_text:
            question_text = question_text.replace("{concept}", random.choice(concepts) if concepts else "concept")
        
        # Generate answers
        correct = self._generate_correct_answer(template["type"])
        incorrect = self._generate_incorrect_answers(correct, 3)
        all_answers = [correct] + incorrect
        random.shuffle(all_answers)
        
        return Question(
            text=question_text,
            answers=all_answers,
            correct_answer=correct,
            difficulty=difficulty,
            points=self._calculate_points(difficulty),
            hint=f"Think about {random.choice(concepts) if concepts else 'the basics'}",
            explanation=f"The correct answer is {correct} because it relates to {self.topic}"
        )
    
    def _generate_correct_answer(self, q_type: str) -> str:
        """Generate correct answer based on question type"""
        if q_type == "symbol":
            symbols = {"Hydrogen": "H", "Carbon": "C", "Oxygen": "O", "Nitrogen": "N"}
            return random.choice(list(symbols.values()))
        elif q_type == "date":
            return str(random.randint(1600, 2023))
        elif q_type == "boolean":
            return random.choice(["True", "False"])
        else:
            return f"Correct answer about {self.topic}"
    
    def _generate_incorrect_answers(self, correct: str, count: int) -> List[str]:
        """Generate plausible incorrect answers"""
        incorrect = []
        
        if correct.isdigit():
            # For dates/numbers
            base = int(correct)
            for _ in range(count):
                offset = random.randint(-50, 50)
                if offset != 0:
                    incorrect.append(str(base + offset))
        elif correct in ["True", "False"]:
            incorrect = ["False"] if correct == "True" else ["True"]
            incorrect.extend([f"Maybe", f"Sometimes"])
        else:
            # For text answers
            variations = [
                f"Incorrect: {self.topic}",
                f"Not quite: {random.choice(self.analysis['related_terms']) if self.analysis['related_terms'] else 'option'}",
                f"Alternative: {random.choice(['A', 'B', 'C', 'D'])}",
                f"Different: {random.choice(['X', 'Y', 'Z'])}"
            ]
            incorrect = random.sample(variations, min(count, len(variations)))
        
        return incorrect[:count]
    
    def _calculate_points(self, difficulty: GameDifficulty) -> int:
        """Calculate points based on difficulty"""
        return {
            GameDifficulty.EASY: 5,
            GameDifficulty.MEDIUM: 10,
            GameDifficulty.HARD: 20,
            GameDifficulty.EXPERT: 50
        }.get(difficulty, 10)

#GAME ENGINES 
class BaseGame(metaclass=GameMeta):
    """Abstract base class for all games"""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.content_generator = SmartContentGenerator(config.topic)
        self.questions: List[Question] = []
        self.game_state: Dict[str, Any] = {}
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Setup game-specific initialization"""
        pass
    
    @abstractmethod
    def generate_html(self) -> str:
        """Generate complete HTML game file"""
        pass
    
    def _generate_base_html(self, body_content: str, additional_js: str = "", additional_css: str = "") -> str:
        """Generate base HTML structure"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title} - {self.config.mode.value.title()} Game</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, {self.config.theme_color} 0%, #667eea 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        
        .game-container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
            max-width: 800px;
            width: 100%;
            animation: slideIn 0.5s ease;
        }}
        
        @keyframes slideIn {{
            from {{
                transform: translateY(-30px);
                opacity: 0;
            }}
            to {{
                transform: translateY(0);
                opacity: 1;
            }}
        }}
        
        .game-header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: {self.config.theme_color};
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .score-board {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 10px;
        }}
        
        .score-item {{
            text-align: center;
        }}
        
        .score-label {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .score-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: {self.config.theme_color};
        }}
        
        .btn {{
            background: {self.config.theme_color};
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            margin: 5px;
        }}
        
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        .btn:active {{
            transform: translateY(0);
        }}
        
        .question-container {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            margin: 20px 0;
        }}
        
        .question-text {{
            font-size: 1.3em;
            color: #333;
            margin-bottom: 20px;
            line-height: 1.5;
        }}
        
        .answers-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .answer-btn {{
            padding: 15px;
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 16px;
        }}
        
        .answer-btn:hover {{
            border-color: {self.config.theme_color};
            background: rgba(74, 144, 226, 0.1);
        }}
        
        .answer-btn.selected {{
            border-color: {self.config.theme_color};
            background: {self.config.theme_color};
            color: white;
        }}
        
        .answer-btn.correct {{
            background: #4caf50;
            border-color: #4caf50;
            color: white;
            animation: pulse 0.5s;
        }}
        
        .answer-btn.incorrect {{
            background: #f44336;
            border-color: #f44336;
            color: white;
            animation: shake 0.5s;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        @keyframes shake {{
            0%, 100% {{ transform: translateX(0); }}
            25% {{ transform: translateX(-5px); }}
            75% {{ transform: translateX(5px); }}
        }}
        
        .progress-bar {{
            width: 100%;
            height: 10px;
            background: #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
            margin: 20px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: {self.config.theme_color};
            transition: width 0.3s ease;
        }}
        
        .hint-box {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            display: none;
        }}
        
        .hint-box.show {{
            display: block;
            animation: slideIn 0.3s;
        }}
        
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        
        .modal.show {{
            display: flex;
        }}
        
        .modal-content {{
            background: white;
            padding: 30px;
            border-radius: 20px;
            max-width: 500px;
            text-align: center;
            animation: slideIn 0.3s;
        }}
        
        {additional_css}
        {self.config.custom_css}
    </style>
</head>
<body>
    {body_content}
    
    <script>
        // Game State Management
        class GameState {{
            constructor() {{
                this.score = 0;
                this.currentQuestion = 0;
                this.totalQuestions = {self.config.question_count};
                this.correctAnswers = 0;
                this.startTime = Date.now();
                this.answers = [];
                this.hintsUsed = 0;
            }}
            
            updateScore(points) {{
                this.score += points;
                this.updateDisplay();
            }}
            
            nextQuestion() {{
                this.currentQuestion++;
                this.updateDisplay();
            }}
            
            updateDisplay() {{
                document.getElementById('score').textContent = this.score;
                document.getElementById('current-question').textContent = this.currentQuestion + 1;
                document.getElementById('total-questions').textContent = this.totalQuestions;
                
                const progress = ((this.currentQuestion + 1) / this.totalQuestions) * 100;
                document.querySelector('.progress-fill').style.width = progress + '%';
            }}
            
            getTimeElapsed() {{
                return Math.floor((Date.now() - this.startTime) / 1000);
            }}
        }}
        
        // Sound Manager
        class SoundManager {{
            constructor(enabled = {str(self.config.enable_sound).lower()}) {{
                this.enabled = enabled;
                this.sounds = {{
                    correct: this.createOscillator(523.25, 0.1), // C5
                    incorrect: this.createOscillator(261.63, 0.2), // C4
                    complete: this.createOscillator(783.99, 0.3) // G5
                }};
            }}
            
            createOscillator(frequency, duration) {{
                return () => {{
                    if (!this.enabled) return;
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const oscillator = audioContext.createOscillator();
                    const gainNode = audioContext.createGain();
                    
                    oscillator.connect(gainNode);
                    gainNode.connect(audioContext.destination);
                    
                    oscillator.frequency.value = frequency;
                    oscillator.type = 'sine';
                    
                    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + duration);
                    
                    oscillator.start(audioContext.currentTime);
                    oscillator.stop(audioContext.currentTime + duration);
                }};
            }}
            
            play(soundType) {{
                if (this.sounds[soundType]) {{
                    this.sounds[soundType]();
                }}
            }}
        }}
        
        // Animation Manager
        class AnimationManager {{
            static fadeIn(element, duration = 300) {{
                element.style.opacity = '0';
                element.style.display = 'block';
                
                const start = performance.now();
                
                requestAnimationFrame(function animate(time) {{
                    const elapsed = time - start;
                    const progress = Math.min(elapsed / duration, 1);
                    
                    element.style.opacity = progress;
                    
                    if (progress < 1) {{
                        requestAnimationFrame(animate);
                    }}
                }});
            }}
            
            static shake(element) {{
                element.classList.add('shake');
                setTimeout(() => element.classList.remove('shake'), 500);
            }}
            
            static pulse(element) {{
                element.classList.add('pulse');
                setTimeout(() => element.classList.remove('pulse'), 500);
            }}
        }}
        
        // Initialize game
        const gameState = new GameState();
        const soundManager = new SoundManager();
        
        {additional_js}
        {self.config.custom_js}
    </script>
</body>
</html>"""

class QuizGame(BaseGame):
    """Interactive Quiz Game"""
    
    def _setup(self):
        self.questions = self.content_generator.generate_questions(
            self.config.question_count,
            self.config.difficulty
        )
    
    def generate_html(self) -> str:
        """Generate quiz game HTML"""
        questions_json = json.dumps([{
            'id': q.id,
            'text': q.text,
            'answers': q.answers,
            'correct': q.correct_answer,
            'hint': q.hint,
            'explanation': q.explanation,
            'points': q.points
        } for q in self.questions])
        
        body_content = f"""
        <div class="game-container">
            <div class="game-header">
                <h1>{self.config.title}</h1>
                <p>Test your knowledge about {self.config.topic}!</p>
            </div>
            
            <div class="score-board">
                <div class="score-item">
                    <div class="score-label">Score</div>
                    <div class="score-value" id="score">0</div>
                </div>
                <div class="score-item">
                    <div class="score-label">Question</div>
                    <div class="score-value">
                        <span id="current-question">1</span> / <span id="total-questions">{self.config.question_count}</span>
                    </div>
                </div>
                <div class="score-item">
                    <div class="score-label">Time</div>
                    <div class="score-value" id="timer">00:00</div>
                </div>
            </div>
            
            <div class="progress-bar">
                <div class="progress-fill"></div>
            </div>
            
            <div id="question-area">
                <div class="question-container">
                    <div class="question-text" id="question-text"></div>
                    <div class="answers-grid" id="answers-grid"></div>
                </div>
                
                <div class="hint-box" id="hint-box"></div>
                
                <div style="text-align: center; margin-top: 20px;">
                    <button class="btn" onclick="showHint()" id="hint-btn">💡 Show Hint</button>
                    <button class="btn" onclick="submitAnswer()" id="submit-btn">Submit Answer</button>
                    <button class="btn" onclick="nextQuestion()" id="next-btn" style="display: none;">Next Question →</button>
                </div>
            </div>
            
            <div class="modal" id="game-over-modal">
                <div class="modal-content">
                    <h2>🎉 Game Complete!</h2>
                    <p style="font-size: 1.2em; margin: 20px 0;">
                        Final Score: <strong id="final-score">0</strong>
                    </p>
                    <p>Correct Answers: <span id="correct-count">0</span> / {self.config.question_count}</p>
                    <p>Time: <span id="final-time">00:00</span></p>
                    <p>Accuracy: <span id="accuracy">0</span>%</p>
                    <button class="btn" onclick="restartGame()">Play Again</button>
                </div>
            </div>
        </div>
        """
        
        additional_js = f"""
        const questions = {questions_json};
        let selectedAnswer = null;
        let answered = false;
        
        function loadQuestion() {{
            if (gameState.currentQuestion >= questions.length) {{
                endGame();
                return;
            }}
            
            const question = questions[gameState.currentQuestion];
            document.getElementById('question-text').textContent = question.text;
            
            const answersGrid = document.getElementById('answers-grid');
            answersGrid.innerHTML = '';
            
            question.answers.forEach((answer, index) => {{
                const btn = document.createElement('button');
                btn.className = 'answer-btn';
                btn.textContent = answer;
                btn.onclick = () => selectAnswer(answer, btn);
                answersGrid.appendChild(btn);
            }});
            
            // Reset UI
            selectedAnswer = null;
            answered = false;
            document.getElementById('submit-btn').style.display = 'inline-block';
            document.getElementById('next-btn').style.display = 'none';
            document.getElementById('hint-box').classList.remove('show');
            document.getElementById('hint-btn').disabled = false;
            
            gameState.updateDisplay();
        }}
        
        function selectAnswer(answer, btn) {{
            if (answered) return;
            
            document.querySelectorAll('.answer-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            selectedAnswer = answer;
        }}
        
        function submitAnswer() {{
            if (!selectedAnswer || answered) return;
            
            answered = true;
            const question = questions[gameState.currentQuestion];
            const correct = selectedAnswer === question.correct;
            
            document.querySelectorAll('.answer-btn').forEach(btn => {{
                if (btn.textContent === question.correct) {{
                    btn.classList.add('correct');
                }} else if (btn.textContent === selectedAnswer) {{
                    btn.classList.add('incorrect');
                }}
            }});
            
            if (correct) {{
                gameState.correctAnswers++;
                gameState.updateScore(question.points);
                soundManager.play('correct');
            }} else {{
                soundManager.play('incorrect');
            }}
            
            if (question.explanation) {{
                document.getElementById('hint-box').innerHTML = 
                    '<strong>Explanation:</strong> ' + question.explanation;
                document.getElementById('hint-box').classList.add('show');
            }}
            
            document.getElementById('submit-btn').style.display = 'none';
            document.getElementById('next-btn').style.display = 'inline-block';
            document.getElementById('hint-btn').disabled = true;
        }}
        
        function nextQuestion() {{
            gameState.nextQuestion();
            loadQuestion();
        }}
        
        function showHint() {{
            const question = questions[gameState.currentQuestion];
            if (question.hint && !answered) {{
                document.getElementById('hint-box').textContent = '💡 Hint: ' + question.hint;
                document.getElementById('hint-box').classList.add('show');
                gameState.hintsUsed++;
                document.getElementById('hint-btn').disabled = true;
            }}
        }}
        
        function endGame() {{
            const accuracy = Math.round((gameState.correctAnswers / gameState.totalQuestions) * 100);
            document.getElementById('final-score').textContent = gameState.score;
            document.getElementById('correct-count').textContent = gameState.correctAnswers;
            document.getElementById('accuracy').textContent = accuracy;
            document.getElementById('final-time').textContent = formatTime(gameState.getTimeElapsed());
            document.getElementById('game-over-modal').classList.add('show');
            soundManager.play('complete');
        }}
        
        function restartGame() {{
            location.reload();
        }}
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        // Update timer
        setInterval(() => {{
            document.getElementById('timer').textContent = formatTime(gameState.getTimeElapsed());
        }}, 1000);
        
        // Start game
        window.onload = () => {{
            loadQuestion();
        }};
        """
        
        return self._generate_base_html(body_content, additional_js)

class MemoryGame(BaseGame):
    """Memory Card Matching Game"""
    
    def _setup(self):
        self.questions = self.content_generator.generate_questions(
            self.config.question_count // 2,  # Half for pairs
            self.config.difficulty
        )
    
    def generate_html(self) -> str:
        """Generate memory game HTML"""
        # Create card pairs from questions
        cards = []
        for q in self.questions:
            cards.append({'id': q.id + '_q', 'content': q.text, 'match': q.id})
            cards.append({'id': q.id + '_a', 'content': q.correct_answer, 'match': q.id})
        
        cards_json = json.dumps(cards)
        
        body_content = f"""
        <div class="game-container">
            <div class="game-header">
                <h1>{self.config.title}</h1>
                <p>Match the cards about {self.config.topic}!</p>
            </div>
            
            <div class="score-board">
                <div class="score-item">
                    <div class="score-label">Matches</div>
                    <div class="score-value" id="matches">0</div>
                </div>
                <div class="score-item">
                    <div class="score-label">Moves</div>
                    <div class="score-value" id="moves">0</div>
                </div>
                <div class="score-item">
                    <div class="score-label">Time</div>
                    <div class="score-value" id="timer">00:00</div>
                </div>
            </div>
            
            <div id="game-board" class="memory-board"></div>
            
            <div style="text-align: center; margin-top: 20px;">
                <button class="btn" onclick="shuffleCards()">🔀 Shuffle</button>
                <button class="btn" onclick="restartGame()">🔄 Restart</button>
            </div>
        </div>
        """
        
        additional_css = """
        .memory-board {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
            perspective: 1000px;
        }
        
        .memory-card {
            height: 120px;
            position: relative;
            transform-style: preserve-3d;
            transition: transform 0.6s;
            cursor: pointer;
        }
        
        .memory-card.flipped {
            transform: rotateY(180deg);
        }
        
        .memory-card.matched {
            opacity: 0.6;
            pointer-events: none;
        }
        
        .card-face {
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
            text-align: center;
            font-size: 14px;
        }
        
        .card-front {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 24px;
        }
        
        .card-back {
            background: white;
            border: 2px solid #e0e0e0;
            transform: rotateY(180deg);
        }
        """
        
        additional_js = f"""
        const cards = {cards_json};
        let flippedCards = [];
        let matches = 0;
        let moves = 0;
        let canFlip = true;
        
        function initializeBoard() {{
            const board = document.getElementById('game-board');
            board.innerHTML = '';
            
            // Shuffle cards
            const shuffled = [...cards].sort(() => Math.random() - 0.5);
            
            shuffled.forEach(card => {{
                const cardElement = document.createElement('div');
                cardElement.className = 'memory-card';
                cardElement.dataset.id = card.id;
                cardElement.dataset.match = card.match;
                
                cardElement.innerHTML = `
                    <div class="card-face card-front">?</div>
                    <div class="card-face card-back">${{card.content}}</div>
                `;
                
                cardElement.onclick = () => flipCard(cardElement);
                board.appendChild(cardElement);
            }});
        }}
        
        function flipCard(card) {{
            if (!canFlip || card.classList.contains('flipped') || card.classList.contains('matched')) return;
            
            card.classList.add('flipped');
            flippedCards.push(card);
            
            if (flippedCards.length === 2) {{
                canFlip = false;
                moves++;
                document.getElementById('moves').textContent = moves;
                
                const [card1, card2] = flippedCards;
                const match1 = card1.dataset.match;
                const match2 = card2.dataset.match;
                
                if (match1 === match2 && card1.dataset.id !== card2.dataset.id) {{
                    // Match found!
                    setTimeout(() => {{
                        card1.classList.add('matched');
                        card2.classList.add('matched');
                        matches++;
                        document.getElementById('matches').textContent = matches;
                        soundManager.play('correct');
                        
                        if (matches === cards.length / 2) {{
                            endGame();
                        }}
                    }}, 500);
                    
                    flippedCards = [];
                    canFlip = true;
                }} else {{
                    // No match
                    setTimeout(() => {{
                        card1.classList.remove('flipped');
                        card2.classList.remove('flipped');
                        flippedCards = [];
                        canFlip = true;
                    }}, 1000);
                }}
            }}
        }}
        
        function shuffleCards() {{
            initializeBoard();
        }}
        
        function endGame() {{
            alert(`Congratulations! You found all matches in ${{moves}} moves!`);
            soundManager.play('complete');
        }}
        
        // Timer
        setInterval(() => {{
            document.getElementById('timer').textContent = formatTime(gameState.getTimeElapsed());
        }}, 1000);
        
        function formatTime(seconds) {{
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
        }}
        
        // Initialize
        window.onload = () => {{
            initializeBoard();
        }};
        """
        
        return self._generate_base_html(body_content, additional_js, additional_css)

# GAME FACTORY 
class GameFactory:
    """Factory for creating games"""
    
    @staticmethod
    def create_game(config: GameConfig) -> BaseGame:
        """Create a game based on configuration"""
        game_map = {
            GameMode.QUIZ: QuizGame,
            GameMode.MEMORY: MemoryGame,
            # Add more game types here
        }
        
        game_class = game_map.get(config.mode)
        if not game_class:
            raise ValueError(f"Game mode {config.mode} not implemented")
        
        return game_class(config)
    
    @staticmethod
    @performance_monitor
    async def create_game_async(config: GameConfig) -> BaseGame:
        """Async game creation with performance monitoring"""
        return GameFactory.create_game(config)

# GAME GENERATOR
class AdvancedGameGenerator:
    """Main class for generating educational games"""
    
    def __init__(self, output_dir: str = "generated_games"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    @contextmanager
    def _file_manager(self, filename: str):
        """Context manager for file operations"""
        filepath = self.output_dir / filename
        file = None
        try:
            file = open(filepath, 'w', encoding='utf-8')
            yield file
            logger.info(f"Successfully created: {filepath}")
        except Exception as e:
            logger.error(f"Error creating file {filename}: {e}")
            raise
        finally:
            if file:
                file.close()
    
    def generate_game(
        self,
        topic: str,
        title: Optional[str] = None,
        mode: GameMode = GameMode.QUIZ,
        difficulty: GameDifficulty = GameDifficulty.MEDIUM,
        **kwargs
    ) -> str:
        """Generate a game from a topic"""
        
        # Create configuration
        config = GameConfig(
            title=title or f"{topic.title()} Challenge",
            topic=topic,
            mode=mode,
            difficulty=difficulty,
            **kwargs
        )
        
        # Create game
        game = GameFactory.create_game(config)
        
        # Generate HTML
        html_content = game.generate_html()
        
        # Save to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{topic.lower().replace(' ', '_')}_{mode.value}_{timestamp}.html"
        
        with self._file_manager(filename) as file:
            file.write(html_content)
        
        return str(self.output_dir / filename)
    
    async def generate_multiple_games(
        self,
        topics: List[str],
        modes: List[GameMode] = None
    ) -> List[str]:
        """Generate multiple games asynchronously"""
        if modes is None:
            modes = [GameMode.QUIZ, GameMode.MEMORY]
        
        tasks = []
        for topic in topics:
            for mode in modes:
                config = GameConfig(
                    title=f"{topic.title()} {mode.value.title()}",
                    topic=topic,
                    mode=mode
                )
                tasks.append(GameFactory.create_game_async(config))
        
        games = await asyncio.gather(*tasks)
        
        files = []
        for game in games:
            html_content = game.generate_html()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{game.config.topic.lower().replace(' ', '_')}_{game.config.mode.value}_{timestamp}.html"
            
            with self._file_manager(filename) as file:
                file.write(html_content)
            
            files.append(str(self.output_dir / filename))
        
        return files

# MAIN EXECUTION 
def main():
    """Main execution function"""
    generator = AdvancedGameGenerator()
    
    # Example: Generate a chemistry quiz game
    chemistry_quiz = generator.generate_game(
        topic="Chemistry and Moles",
        title="Chemistry Happens in Moles!",
        mode=GameMode.QUIZ,
        difficulty=GameDifficulty.MEDIUM,
        question_count=10,
        theme_color="#0EA5E9"
    )
    print(f"Created chemistry quiz: {chemistry_quiz}")
    
    # Example: Generate a memory game
    memory_game = generator.generate_game(
        topic="Periodic Table Elements",
        title="Element Memory Match",
        mode=GameMode.MEMORY,
        difficulty=GameDifficulty.EASY,
        question_count=8,
        theme_color="#10B981"
    )
    print(f"Created memory game: {memory_game}")
    
    # Generate multiple games asynchronously
    async def generate_batch():
        topics = ["Mathematics", "Physics", "Biology", "History"]
        files = await generator.generate_multiple_games(topics, [GameMode.QUIZ])
        for file in files:
            print(f"Generated: {file}")
    
    # Run async generation
    # asyncio.run(generate_batch())
    
    print("\n🎮 All games generated successfully!")
    print(f"📁 Games saved to: {generator.output_dir}")

if __name__ == "__main__":
    main()

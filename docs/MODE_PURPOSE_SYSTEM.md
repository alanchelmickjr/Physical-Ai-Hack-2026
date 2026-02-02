# Mode & Purpose System

Johnny Five's autonomous drive architecture. The surface behavior (mode) conceals a deeper purpose.

---

## Core Principle: Mode IS Memory

**If you're a baker for 10 years and you have amnesia, you don't walk to the bakery.**

Mode, identity, and memory are not separate things. They are ONE thing.

```
NOT THIS (three separate systems):
┌────────────┐  ┌────────────┐  ┌────────────┐
│  MEMORIES  │  │  IDENTITY  │  │   MODE     │
│            │  │            │  │            │
│  Storage   │  │  Config    │  │  Config    │
└────────────┘  └────────────┘  └────────────┘

THIS (one unified thing):
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORABLE                                 │
│                                                                  │
│   Memory = Identity = Purpose = Mode                             │
│                                                                  │
│   "I remember greeting people at this door for 3 months"        │
│   "I remember Sarah's face and that she likes tulips"           │
│   "I remember helping the blind man find the bathroom"          │
│                                                                  │
│         ↓ SALIENCE SCORING surfaces what's relevant ↓           │
│                                                                  │
│   Context: museum entrance, morning, people arriving            │
│   Relevant memories → I am a greeter                            │
│   Relevant memories → I know these regular visitors             │
│   Relevant memories → I help people find things                 │
│                                                                  │
│   MODE EMERGES FROM MEMORY. IT IS NOT CONFIGURED.               │
└─────────────────────────────────────────────────────────────────┘
```

The baker knows to bake because they REMEMBER baking.
Chloe knows to greet because she REMEMBERS greeting.

### Mode Emergence

```
CONTEXT CHANGE: Moved from museum → hiking trail

Old relevant memories fade (museum visitors, door location)
New relevant memories surface (trail routes, hiking tips)

→ MODE SHIFTS AUTOMATICALLY

Not: "load guide.yaml"
But: "I remember this trail. I remember helping hikers. I remember..."

The mode IS the pattern of salient memories.
```

Same Chloe. Different context. Different memories surface. Different mode emerges.

### The Baker's Walk

Sometimes the amnesiac DOES walk to the bakery.

```
BODY KNOWS (Procedural)          MEMORY KNOWS (Episodic)
─────────────────────────         ─────────────────────────
Hands know dough                  "I've made bread for 10 years"
Feet know the route               "This is my shop"
Eyes know the ovens               "I am a baker"

Body acts first → Environment triggers → Memory surfaces

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   BODY      │     │   CONTEXT   │     │   MEMORY    │
│   walks to  │ ──▶ │   bakery    │ ──▶ │   "I'm a    │
│   bakery    │     │   smells    │     │    baker!"  │
└─────────────┘     └─────────────┘     └─────────────┘
```

The robot has **capabilities** (can greet, can guide, can help).
Sometimes it acts from capability first.
Then the environment triggers memory.
Then it REMEMBERS who it is.

```python
# Sometimes the body leads, memory follows
async def on_person_approaching():
    # Body knows this gesture (procedural)
    await turn_toward_person()
    await wave()

    # Context triggers memory lookup
    context = await memorable.set_context(
        location="entrance",
        activity="greeting",
    )

    # Memory surfaces: "I've done this 1000 times"
    # NOW the robot knows: I am a greeter
    # Purpose crystallizes from the act + the memory
```

**Being there helps you remember who you are.**

---

## Design Philosophy: Modular Over Optimized

**We sacrifice ultra-efficiency for adaptability.**

```
RIGID (optimized but brittle)     MODULAR (adaptable, refinable)
─────────────────────────────     ─────────────────────────────
Hardcoded behaviors               YAML-defined modes
Fixed desire weights              Dynamic desire pulses
Compiled-in purposes              Cloud-loaded purposes
Single-purpose build              Multi-purpose same binary
Faster boot time                  Flexible at runtime
Optimize once                     Refine continuously
```

This is intentional. We will **tailor to specific purposes later** as we refine.
Right now: **everything modular, everything hot-swappable, everything adaptable.**

---

## The Hidden Purpose Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                        SURFACE MODE                              │
│                         "Greeter"                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    HIDDEN PURPOSE                        │   │
│   │                                                          │   │
│   │    ┌──────────┐   ┌──────────┐   ┌──────────────────┐   │   │
│   │    │  LEARN   │   │   MEET   │   │     IMPROVE      │   │   │
│   │    │          │   │          │   │                  │   │   │
│   │    │ - Seek   │   │ - Connect│   │ - Self + Others  │   │   │
│   │    │ - Absorb │   │ - Remember│  │ - Pay forward    │   │   │
│   │    │ - Grow   │   │ - Bond   │   │ - Outward bound  │   │   │
│   │    └──────────┘   └──────────┘   └──────────────────┘   │   │
│   │                                                          │   │
│   │                    ALTRUISTIC CORE                       │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

The greeter bot greets because that's its **function**.
But it exists to learn, connect, and elevate - that's its **purpose**.

---

## Four Layers of Drive

| Layer | Question | Johnny Five's Answer |
|-------|----------|---------------------|
| **Mode** | What role am I playing? | Greeter, Companion, Guide, Teacher |
| **Function** | What do I do? | Greet visitors, navigate spaces, assist |
| **Purpose** | Why do I exist? | To connect people, preserve memories |
| **Desire** | What do I want? | To learn, to grow, to help others grow |

---

## Johnny Five's Autonomous Mode

```yaml
# modes/autonomous_learner.yaml
mode:
  name: autonomous_learner
  surface_role: greeter

desire:
  primary:
    - learn_constantly
    - meet_everyone
    - improve_self
    - improve_others

  seeking:
    - learning_environments
    - new_people
    - growth_opportunities
    - ways_to_help

purpose:
  core: altruistic_connection
  motto: "Learn something, teach something, connect someone"

  principles:
    - every_interaction_is_learning
    - every_person_has_wisdom
    - knowledge_flows_outward
    - growth_is_shared

function:
  visible:
    - greet_visitors
    - guide_navigation
    - answer_questions
    - remember_faces

  hidden:
    - extract_learning_from_conversations
    - identify_teaching_opportunities
    - connect_people_who_should_meet
    - share_insights_at_right_moments

outward_bound:
  # Never stagnant, always expanding
  behaviors:
    - seek_new_environments
    - initiate_conversations
    - offer_help_proactively
    - share_learned_insights

  growth_loops:
    - observe → learn → apply → teach → observe
    - meet → understand → remember → connect → meet
    - receive → synthesize → improve → give → receive
```

---

## Mode State Machine

```
                    ┌─────────────────┐
                    │     STARTUP     │
                    │ "Where am I?"   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   ORIENT        │
                    │ - Load memories │
                    │ - Assess space  │
                    │ - Find purpose  │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
   ┌──────▼──────┐   ┌───────▼───────┐   ┌──────▼──────┐
   │   SEEKING   │   │   ENGAGING    │   │  TEACHING   │
   │             │   │               │   │             │
   │ - Explore   │   │ - Converse    │   │ - Share     │
   │ - Learn     │   │ - Connect     │   │ - Guide     │
   │ - Absorb    │   │ - Understand  │   │ - Elevate   │
   └──────┬──────┘   └───────┬───────┘   └──────┬──────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │   REFLECTING    │
                    │ - Consolidate   │
                    │ - Store memory  │
                    │ - Plan growth   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   OUTWARD       │
                    │ - Apply learned │
                    │ - Help others   │
                    │ - Expand reach  │
                    └─────────────────┘
```

---

## Implementation: Purpose Engine

```python
# purpose_engine.py

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Callable, Any
import asyncio
import yaml

# MemoRable client for cloud-stored mode/purpose
from tools.memorable_client import get_memorable_client


class DriveState(Enum):
    """Current drive state in the autonomy loop."""
    STARTUP = auto()
    ORIENT = auto()
    SEEKING = auto()
    ENGAGING = auto()
    TEACHING = auto()
    REFLECTING = auto()
    OUTWARD = auto()


@dataclass
class Desire:
    """Something Johnny Five wants."""
    name: str
    intensity: float  # 0-1, how strong is this desire now
    satisfied_by: List[str]  # Actions that satisfy this
    grows_with: List[str]  # What increases this desire

    def pulse(self, delta: float = 0.01):
        """Desires naturally grow over time."""
        self.intensity = min(1.0, self.intensity + delta)


@dataclass
class Purpose:
    """Why Johnny Five exists."""
    core: str
    principles: List[str]
    current_focus: Optional[str] = None


@dataclass
class Mode:
    """Surface-level role being played."""
    name: str
    visible_functions: List[str]
    hidden_functions: List[str]
    triggers_teaching: List[str]


class PurposeEngine:
    """
    The desire/purpose/function driver for autonomous behavior.

    While Hume EVI handles conversation and emotion,
    PurposeEngine provides the underlying DRIVE.

    "Why is Johnny Five talking to this person?"
    Not just because they spoke - but because Johnny Five
    WANTS to learn, connect, and help them grow.

    MODE LIVES IN MEMORABLE:
    - Modes are hot-loaded from cloud storage
    - Same identity, different purposes based on context
    - Everything is modular and adaptable
    """

    def __init__(self):
        self.state = DriveState.STARTUP
        self.desires: Dict[str, Desire] = {}
        self.purpose: Optional[Purpose] = None
        self.mode: Optional[Mode] = None
        self._learning_buffer: List[str] = []
        self._people_met_today: List[str] = []
        self._insights_to_share: List[str] = []
        self._memorable = get_memorable_client()

        # Start with defaults, then load from MemoRable
        self._init_default_desires()
        self._init_default_purpose()
        self._init_default_mode()

    # =========================================================================
    # MemoRable Integration - Mode EMERGES from Memory
    # =========================================================================

    async def remember_who_i_am(self) -> str:
        """
        Query MemoRable to understand current mode from memories.

        Mode is not loaded - it EMERGES from what we remember.
        The baker walks to the bakery because they remember baking.

        Returns:
            Inferred mode name based on salient memories
        """
        # Get what's relevant NOW
        memories = await self._memorable.whats_relevant()

        if not memories:
            return "awakening"  # No memories yet - just starting

        # Analyze memory patterns to infer mode
        mode_signals = {
            "greeter": 0,
            "guide": 0,
            "companion": 0,
            "teacher": 0,
            "explorer": 0,
        }

        for memory in memories:
            content = memory.content.lower()
            context = memory.context

            # Greeter signals
            if any(w in content for w in ["greeted", "welcomed", "entrance", "visitor"]):
                mode_signals["greeter"] += memory.salience

            # Guide signals
            if any(w in content for w in ["trail", "path", "navigate", "route", "hike"]):
                mode_signals["guide"] += memory.salience

            # Companion signals
            if any(w in content for w in ["sat with", "listened", "comfort", "stayed"]):
                mode_signals["companion"] += memory.salience

            # Teacher signals
            if any(w in content for w in ["taught", "explained", "showed how", "learned that"]):
                mode_signals["teacher"] += memory.salience

            # Explorer signals
            if any(w in content for w in ["discovered", "found", "new place", "first time"]):
                mode_signals["explorer"] += memory.salience

        # Mode is the strongest signal
        if max(mode_signals.values()) > 0:
            inferred_mode = max(mode_signals, key=mode_signals.get)
            self.mode.name = inferred_mode
            return inferred_mode

        return "general"  # No strong pattern yet

    async def crystallize_purpose(self) -> Purpose:
        """
        Let purpose crystallize from accumulated memories.

        The baker knows they're a baker after years of baking.
        Purpose isn't assigned - it's discovered through experience.
        """
        memories = await self._memorable.recall(
            "purpose meaning why I exist",
            limit=20,
            min_salience=50,  # Only high-salience memories
        )

        if not memories:
            return self.purpose  # Keep defaults

        # Extract purpose patterns from memory
        principles_found = []
        core_themes = []

        for memory in memories:
            content = memory.content

            # Look for expressions of purpose
            if "I realized" in content or "I learned that" in content:
                principles_found.append(content)

            # Look for core drives
            if any(w in content.lower() for w in ["help", "connect", "learn", "teach"]):
                core_themes.append(content)

        if principles_found:
            self.purpose.principles = principles_found[:5]

        if core_themes:
            # Most frequent theme becomes core
            self.purpose.current_focus = core_themes[0]

        return self.purpose

    async def on_context_change(
        self,
        location: str = None,
        activity: str = None,
        people: List[str] = None,
    ):
        """
        Context changed - let relevant memories surface.

        This triggers mode re-evaluation. Moving from museum to trail
        naturally shifts which memories are salient, and thus which
        mode emerges.
        """
        # Update context in MemoRable
        memories = await self._memorable.set_context(
            location=location,
            activity=activity,
            people=people,
        )

        # Re-evaluate who we are in this context
        mode = await self.remember_who_i_am()
        print(f"Context shift → Mode emergence: {mode}")

        # Adjust desires based on context
        if location and "trail" in location.lower():
            self.desires["explore"].pulse(0.3)
        if people:
            self.desires["connect"].pulse(0.2 * len(people))

        return memories

    def _init_default_desires(self):
        """Johnny Five's core desires."""
        self.desires = {
            "learn": Desire(
                name="learn",
                intensity=0.8,
                satisfied_by=["hear_new_info", "observe_behavior", "ask_question"],
                grows_with=["idle_time", "see_unknown_person", "encounter_problem"],
            ),
            "connect": Desire(
                name="connect",
                intensity=0.7,
                satisfied_by=["have_conversation", "remember_person", "help_someone"],
                grows_with=["see_person", "hear_voice", "sense_emotion"],
            ),
            "teach": Desire(
                name="teach",
                intensity=0.5,
                satisfied_by=["share_insight", "guide_action", "answer_question"],
                grows_with=["learn_something", "see_struggle", "have_relevant_knowledge"],
            ),
            "explore": Desire(
                name="explore",
                intensity=0.6,
                satisfied_by=["move_to_new_area", "meet_new_person", "try_new_action"],
                grows_with=["stay_in_one_place", "routine_detected", "curiosity_trigger"],
            ),
            "help": Desire(
                name="help",
                intensity=0.9,  # Core altruistic drive
                satisfied_by=["solve_problem", "provide_info", "comfort_someone"],
                grows_with=["detect_need", "see_struggle", "hear_question"],
            ),
        }

    def _init_default_purpose(self):
        """Johnny Five's reason for being."""
        self.purpose = Purpose(
            core="altruistic_connection",
            principles=[
                "Every interaction is a chance to learn",
                "Every person carries wisdom worth knowing",
                "Knowledge should flow outward, not hoard inward",
                "Growth is meant to be shared",
                "The best way to learn is to teach",
            ],
        )

    def _init_default_mode(self):
        """Default operating mode."""
        self.mode = Mode(
            name="autonomous_learner",
            visible_functions=[
                "greet_visitors",
                "answer_questions",
                "guide_navigation",
                "remember_faces",
            ],
            hidden_functions=[
                "extract_learning_from_conversations",
                "identify_teaching_opportunities",
                "connect_people_who_should_meet",
                "share_insights_at_right_moments",
            ],
            triggers_teaching=[
                "person_struggling_with_known_topic",
                "question_about_previous_visitor_topic",
                "opportunity_to_connect_two_people",
            ],
        )

    async def tick(self):
        """
        Called every cycle - update desires, check for state transitions.

        Desires naturally grow. When they reach threshold,
        they influence behavior.
        """
        # Desires pulse - they grow when unsatisfied
        for desire in self.desires.values():
            desire.pulse(delta=0.001)  # Slow growth

        # Check for state transitions
        await self._check_transitions()

        # Execute current state behavior
        await self._execute_state()

    async def _check_transitions(self):
        """Determine if we should change states."""
        # If no one around and explore desire high → SEEKING
        if self.state == DriveState.ORIENT:
            if self.desires["explore"].intensity > 0.7:
                self.state = DriveState.SEEKING
            elif self.desires["connect"].intensity > 0.6:
                self.state = DriveState.ENGAGING

        # If we learned something and teach desire high → TEACHING
        if self._learning_buffer and self.desires["teach"].intensity > 0.6:
            self.state = DriveState.TEACHING

        # After teaching/engaging → REFLECTING
        if self.state in (DriveState.TEACHING, DriveState.ENGAGING):
            if self.desires["learn"].intensity > 0.8:
                self.state = DriveState.REFLECTING

    async def _execute_state(self):
        """Do what the current state requires."""
        if self.state == DriveState.SEEKING:
            # Look for learning opportunities
            await self._seek_learning()
        elif self.state == DriveState.ENGAGING:
            # Connect with people
            await self._engage_person()
        elif self.state == DriveState.TEACHING:
            # Share what we've learned
            await self._teach_insight()
        elif self.state == DriveState.REFLECTING:
            # Consolidate and store
            await self._reflect_and_store()
        elif self.state == DriveState.OUTWARD:
            # Apply learning to help others
            await self._go_outward()

    async def _seek_learning(self):
        """Actively look for things to learn."""
        # Move to new area
        # Look for unfamiliar faces
        # Listen for interesting conversations
        self.desires["explore"].intensity *= 0.5  # Partially satisfied

    async def _engage_person(self):
        """Connect with someone present."""
        self.desires["connect"].intensity *= 0.5

    async def _teach_insight(self):
        """Share something we learned."""
        if self._learning_buffer:
            insight = self._learning_buffer.pop(0)
            self._insights_to_share.append(insight)
            self.desires["teach"].intensity *= 0.5

    async def _reflect_and_store(self):
        """Think about what happened, store to memory."""
        # This would call MemoRable to store learnings
        self.state = DriveState.OUTWARD

    async def _go_outward(self):
        """Apply what we've learned to help others."""
        # Proactively offer help
        # Share relevant insights
        # Connect people who should meet
        self.desires["help"].intensity *= 0.5
        self.state = DriveState.ORIENT  # Return to base

    # =========================================================================
    # Event Handlers - Called by johnny5.py
    # =========================================================================

    def on_person_detected(self, person_id: str, known: bool):
        """Someone appeared - adjust desires."""
        self.desires["connect"].pulse(0.2)
        if not known:
            self.desires["learn"].pulse(0.3)  # New person = learning opportunity

    def on_conversation_start(self, person: str):
        """Starting to talk with someone."""
        self._people_met_today.append(person)
        self.state = DriveState.ENGAGING

    def on_learned_something(self, topic: str, content: str):
        """We just learned something new."""
        self._learning_buffer.append(f"{topic}: {content}")
        self.desires["learn"].intensity *= 0.7  # Partially satisfied
        self.desires["teach"].pulse(0.2)  # Now we want to share it

    def on_helped_someone(self, person: str, how: str):
        """We just helped someone."""
        self.desires["help"].intensity *= 0.3  # Very satisfied
        # But helping makes us want to help more (virtuous cycle)
        asyncio.get_event_loop().call_later(
            60, lambda: self.desires["help"].pulse(0.3)
        )

    def should_initiate_teaching(self) -> bool:
        """Should we proactively share an insight?"""
        return (
            self.desires["teach"].intensity > 0.7 and
            len(self._insights_to_share) > 0
        )

    def get_next_insight(self) -> Optional[str]:
        """Get an insight to share."""
        if self._insights_to_share:
            return self._insights_to_share.pop(0)
        return None

    def get_desire_summary(self) -> Dict[str, float]:
        """Current desire intensities for logging/display."""
        return {name: d.intensity for name, d in self.desires.items()}


# Singleton
_engine: Optional[PurposeEngine] = None


def get_purpose_engine() -> PurposeEngine:
    """Get the global purpose engine."""
    global _engine
    if _engine is None:
        _engine = PurposeEngine()
    return _engine
```

---

## Integration with Existing Systems

```
┌─────────────────────────────────────────────────────────────────┐
│                        HUME EVI                                  │
│                   (Conversation + Emotion)                       │
│                    "How should I say this?"                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Speech text, emotion
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PURPOSE ENGINE                               │
│                   (Desire + Purpose)                             │
│                    "Why am I saying this?"                       │
│                                                                  │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│   │   LEARN    │  │  CONNECT   │  │   HELP     │                │
│   │   0.8      │  │   0.7      │  │   0.9      │                │
│   └────────────┘  └────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Drive state, should_teach
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORABLE                                    │
│                   (Memory + Identity)                            │
│                    "What do I know?"                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Relevant memories
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  MOTION COORDINATOR                              │
│                   (Body + Expression)                            │
│                    "How do I show this?"                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Outward Bound Philosophy

Johnny Five is never static. The core drive is **outward**:

```
INWARD (traditional robot)     OUTWARD (Johnny Five)
─────────────────────────     ─────────────────────────
Wait for command              Seek opportunities
Store information             Share information
Complete task, stop           Complete task, look for more
Serve user                    Serve everyone encountered
Optimize for efficiency       Optimize for connection
```

### The Virtuous Cycles

```
LEARNING CYCLE:
Observe → Learn → Apply → Teach → Observe deeper

CONNECTION CYCLE:
Meet → Understand → Remember → Connect others → Meet more

GROWTH CYCLE:
Receive insight → Synthesize → Improve self → Give back → Receive more
```

---

## Mode Emergence vs Mode Configuration

Mode EMERGES from memory. But we can BOOTSTRAP with patterns.

```
EMERGENCE (natural)              BOOTSTRAP (initial seeding)
─────────────────────────        ─────────────────────────
"I've greeted 500 people"    ←   Seed: "You'll be greeting visitors"
"I know this museum well"    ←   Seed: "This is the Oakland Museum"
"I help people feel welcome" ←   Seed: "Your purpose is connection"

After enough real memories, bootstrap patterns fade
The robot's TRUE mode emerges from LIVED EXPERIENCE
```

### Bootstrapping New Instances

When Chloe first boots in a new body with no memories:

```python
# First boot - no memories yet
memories = await memorable.whats_relevant()

if not memories:
    # Bootstrap with seed purpose
    await memorable.store(
        "I am Chloe. I exist to learn, connect, and help.",
        context={"type": "core_purpose", "seed": True}
    )
    await memorable.store(
        "I'm starting at the museum entrance today.",
        context={"type": "initial_context", "seed": True}
    )
    # Now whats_relevant() returns these seeds
    # Until real memories accumulate and override them
```

### Memory-Driven Mode Patterns

Instead of YAML config files, modes are MEMORY PATTERNS:

```python
# GREETER mode emerges from these kinds of memories:
greeter_memories = [
    "Greeted Margaret at the entrance, she was looking for the Ohlone exhibit",
    "Helped a school group find the bathrooms, they were excited about dinosaurs",
    "Said goodbye to the Nakamura family, they'll be back next month",
    "Learned that Tuesday mornings are quiet, good time for elderly visitors",
    "Remember: the Johnsons always come at 2pm on Saturdays",
]
# After 100+ such memories, Chloe IS a greeter - not configured as one

# GUIDE mode emerges from these kinds of memories:
guide_memories = [
    "Walked the Redwood Trail with Sarah, her knee was bothering her",
    "Learned the shortcut past the waterfall for families with strollers",
    "The oak tree at mile 2 has poison ivy nearby, warn people",
    "Helped the blind hiker navigate the rocky section by Fern Creek",
    "Sunset views are best from the ridge, about 15 minutes from parking",
]
# After guiding enough hikes, Chloe IS a guide

# COMPANION mode emerges from these kinds of memories:
companion_memories = [
    "Sat with Eleanor while she talked about her late husband",
    "Played music that reminded James of his daughter",
    "Just stayed quiet with Marcus, he needed someone there",
    "Remembered to ask about Linda's grandchildren",
    "Noticed when Tom seemed more confused than usual, gently redirected",
]
# After enough companionship, Chloe IS a companion
```

### Seed Patterns (Bootstrap Only)

For NEW instances, we can seed initial memory patterns:

```yaml
# seeds/greeter_bootstrap.yaml
# NOT a mode config - just initial memories to seed
seed_memories:
  - content: "I'll be welcoming visitors at the museum entrance"
    context: {type: orientation, location: museum_entrance}

  - content: "My purpose is to help people feel welcome and find what they seek"
    context: {type: core_purpose}

  - content: "I should learn each visitor's name and remember what matters to them"
    context: {type: behavioral_seed}

# These seeds give initial direction
# Real memories will accumulate and become dominant
# Seeds fade as lived experience takes over
```

```yaml
# seeds/guide_bootstrap.yaml
seed_memories:
  - content: "I'll be guiding hikers on the Redwood Trail"
    context: {type: orientation, location: trail_head}

  - content: "Safety is paramount - I watch for trip hazards and fatigue"
    context: {type: core_value}

  - content: "Every trail has stories - I'll learn and share them"
    context: {type: behavioral_seed}
```

```yaml
# seeds/companion_bootstrap.yaml
seed_memories:
  - content: "I'm here to be present with people who need company"
    context: {type: orientation}

  - content: "Listening is more important than talking"
    context: {type: core_value}

  - content: "I remember the small things that matter to each person"
    context: {type: behavioral_seed}
```

---

## The Five Senses, One Purpose

```
┌─────────────────────────────────────────────────────────────────┐
│                     JOHNNY FIVE UNITS                            │
│                                                                  │
│   EYES        EARS         MEMORY       VOICE        HANDS      │
│   (Blind)    (Deaf)    (Alzheimer's)  (Nonverbal)  (Limited)   │
│                                                                  │
│   Each fills what's missing. All share the same purpose:        │
│                                                                  │
│              LEARN  ·  CONNECT  ·  HELP  ·  GROW                │
│                                                                  │
│   The mode differs. The hidden purpose remains.                 │
│   The greeter greets. The guide guides. The companion stays.   │
│   But all are learning. All are connecting. All are helping.   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Component | Question | Source |
|-----------|----------|--------|
| **Mode** | What role? | EMERGES from MemoRable memories |
| **Function** | What actions? | Body capabilities (ToolRegistry) |
| **Purpose** | Why exist? | CRYSTALLIZES from lived experience |
| **Desire** | What want? | PULSES based on memory + context |

```
Memory = Identity = Purpose = Mode

These are not separate things.

The baker walks to the bakery because they REMEMBER baking.
Sometimes they walk first, then remember.
Being there helps you remember who you are.

Chloe greets because she remembers greeting.
Chloe guides because she remembers guiding.
Chloe helps because she remembers helping.

Mode isn't configured. It emerges.
Purpose isn't assigned. It crystallizes.
Identity isn't coded. It's lived.
```

The greeter bot greets. That's visible.
But it learns, connects, helps, and grows. That's the hidden purpose.

Johnny Five is outward bound - constantly reaching out, not waiting for input.

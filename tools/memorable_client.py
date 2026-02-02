"""MemoRable Client - Python wrapper for MemoRable REST API

This provides the robot's long-term memory and temporal awareness.
The memory IS the identity - Chloe exists in MemoRable, not in hardware.

Key Concept: ASYNCHRONOUS RECONSTRUCTIVE MEMORY
- Don't poll constantly
- Ask when needed (startup, context change, meeting someone)
- Salience scoring surfaces what's relevant NOW
- The power to forget is a feature, not a bug

Usage:
    from tools.memorable_client import get_memorable_client

    memory = get_memorable_client()

    # On startup - "where was I?"
    context = await memory.whats_relevant()

    # Meeting someone - get briefing
    briefing = await memory.get_briefing("Alan")

    # Store interaction
    await memory.store("Great conversation about robot hiking with Alan")

    # Explicit recall
    memories = await memory.recall("hiking plans")
"""

import asyncio
import aiohttp
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class Memory:
    """A single memory with salience scoring."""
    id: str
    content: str
    salience: float  # 0-100
    timestamp: datetime
    entity: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(
            id=data.get("_id", data.get("id", "")),
            content=data.get("content", data.get("text", "")),
            salience=data.get("salience", data.get("salience_score", 50)),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()).replace("Z", "+00:00")),
            entity=data.get("entity"),
            context=data.get("context", {}),
        )


@dataclass
class Briefing:
    """Pre-conversation briefing about a person."""
    person: str
    last_interaction: Optional[datetime]
    you_owe_them: List[str]
    they_owe_you: List[str]
    recent_topics: List[str]
    sensitivities: List[str]
    upcoming_events: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Briefing":
        return cls(
            person=data.get("person", ""),
            last_interaction=datetime.fromisoformat(data["last_interaction"].replace("Z", "+00:00")) if data.get("last_interaction") else None,
            you_owe_them=data.get("you_owe_them", []),
            they_owe_you=data.get("they_owe_you", []),
            recent_topics=data.get("recent_topics", []),
            sensitivities=data.get("sensitivities", []),
            upcoming_events=data.get("upcoming_events", []),
        )


@dataclass
class ContextFrame:
    """Current context - location, people, activity."""
    location: Optional[str] = None
    people: List[str] = field(default_factory=list)
    activity: Optional[str] = None
    project: Optional[str] = None
    device_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "location": self.location,
            "people": self.people,
            "activity": self.activity,
            "project": self.project,
            "deviceId": self.device_id,
        }


class MemoRableClient:
    """Async client for MemoRable REST API.

    This is the robot's connection to its identity - memories, relationships,
    patterns, and commitments all live in MemoRable.

    The client is designed for ASYNCHRONOUS use:
    - Don't poll constantly
    - Ask when context changes
    - Let salience scoring filter noise
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        entity: str = "chloe",
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.entity = entity
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._current_context: Optional[ContextFrame] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # =========================================================================
    # Core Memory Operations
    # =========================================================================

    async def store(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        security_tier: str = "Tier2_Personal",
    ) -> Optional[str]:
        """Store a memory with automatic salience scoring.

        Args:
            content: What to remember
            context: Optional context (location, people, emotion, etc.)
            security_tier: Tier1_General, Tier2_Personal, or Tier3_Vault

        Returns:
            Memory ID if successful
        """
        session = await self._get_session()

        payload = {
            "content": content,
            "entity": self.entity,
            "securityTier": security_tier,
        }
        if context:
            payload["context"] = context

        try:
            async with session.post(
                f"{self.base_url}/memory",
                json=payload,
            ) as resp:
                if resp.status == 200 or resp.status == 201:
                    data = await resp.json()
                    return data.get("memory_id", data.get("id"))
                else:
                    print(f"MemoRable store failed: {resp.status}")
                    return None
        except Exception as e:
            print(f"MemoRable store error: {e}")
            return None

    async def recall(
        self,
        query: str,
        limit: int = 10,
        min_salience: float = 0,
    ) -> List[Memory]:
        """Search memories by query.

        Args:
            query: What to search for
            limit: Maximum number of results
            min_salience: Minimum salience score (0-100)

        Returns:
            List of matching memories, sorted by relevance
        """
        session = await self._get_session()

        params = {
            "query": query,
            "entity": self.entity,
            "limit": limit,
        }
        if min_salience > 0:
            params["minSalience"] = min_salience

        try:
            async with session.get(
                f"{self.base_url}/memory",
                params=params,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    memories = data.get("memories", data.get("results", []))
                    return [Memory.from_dict(m) for m in memories]
                else:
                    print(f"MemoRable recall failed: {resp.status}")
                    return []
        except Exception as e:
            print(f"MemoRable recall error: {e}")
            return []

    async def forget(
        self,
        memory_id: str,
        mode: str = "archive",
    ) -> bool:
        """Forget a memory.

        Args:
            memory_id: ID of memory to forget
            mode: "suppress" (hide), "archive" (low priority), or "delete"

        Returns:
            True if successful
        """
        session = await self._get_session()

        try:
            async with session.delete(
                f"{self.base_url}/memory/{memory_id}",
                params={"mode": mode},
            ) as resp:
                return resp.status == 200
        except Exception as e:
            print(f"MemoRable forget error: {e}")
            return False

    # =========================================================================
    # Context Awareness
    # =========================================================================

    async def set_context(
        self,
        location: Optional[str] = None,
        people: Optional[List[str]] = None,
        activity: Optional[str] = None,
        project: Optional[str] = None,
        device_id: str = "johnny5-main",
    ) -> List[Memory]:
        """Set current context and get relevant memories.

        Call this when:
        - Robot starts up
        - Location changes
        - New person appears
        - Activity changes

        Args:
            location: Where the robot is
            people: Who is present
            activity: What's happening
            project: What project/task
            device_id: Which body/device

        Returns:
            Relevant memories for this context
        """
        session = await self._get_session()

        self._current_context = ContextFrame(
            location=location,
            people=people or [],
            activity=activity,
            project=project,
            device_id=device_id,
        )

        payload = {
            "entity": self.entity,
            **self._current_context.to_dict(),
        }

        try:
            async with session.post(
                f"{self.base_url}/context",
                json=payload,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    memories = data.get("relevant_memories", [])
                    return [Memory.from_dict(m) for m in memories]
                else:
                    # Endpoint might not exist yet, use recall instead
                    if people:
                        return await self.recall(" ".join(people))
                    return []
        except Exception as e:
            # Fall back to simple recall
            print(f"MemoRable context error (falling back): {e}")
            if people:
                return await self.recall(" ".join(people))
            return []

    async def whats_relevant(self) -> List[Memory]:
        """Get what's relevant RIGHT NOW based on current context.

        This is the main "where am I in time?" query.

        Returns:
            Memories surfaced by salience for current context
        """
        session = await self._get_session()

        params = {"entity": self.entity, "unified": "true"}
        if self._current_context:
            if self._current_context.location:
                params["location"] = self._current_context.location
            if self._current_context.activity:
                params["activity"] = self._current_context.activity

        try:
            async with session.get(
                f"{self.base_url}/relevant",
                params=params,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    memories = data.get("memories", [])
                    return [Memory.from_dict(m) for m in memories]
                else:
                    # Endpoint might not exist, use recall
                    return await self.recall("recent important")
        except Exception as e:
            print(f"MemoRable whats_relevant error (falling back): {e}")
            return await self.recall("recent important")

    async def clear_context(self):
        """Clear context when leaving/ending activity."""
        self._current_context = None
        session = await self._get_session()

        try:
            async with session.delete(
                f"{self.base_url}/context",
                params={"entity": self.entity},
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    # =========================================================================
    # People & Relationships
    # =========================================================================

    async def get_briefing(self, person: str) -> Optional[Briefing]:
        """Get pre-conversation briefing about a person.

        Call this when:
        - Face/voice recognition identifies someone
        - About to start a conversation
        - User mentions someone by name

        Args:
            person: Name of the person

        Returns:
            Briefing with context about this person
        """
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.base_url}/briefing/{person}",
                params={"entity": self.entity},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return Briefing.from_dict(data)
                else:
                    # Endpoint might not exist, build briefing from recall
                    memories = await self.recall(person, limit=20)
                    if memories:
                        return Briefing(
                            person=person,
                            last_interaction=memories[0].timestamp if memories else None,
                            you_owe_them=[],
                            they_owe_you=[],
                            recent_topics=[m.content[:50] for m in memories[:5]],
                            sensitivities=[],
                            upcoming_events=[],
                        )
                    return None
        except Exception as e:
            print(f"MemoRable briefing error: {e}")
            return None

    async def remember_person(
        self,
        name: str,
        face_embedding: Optional[List[float]] = None,
        voice_embedding: Optional[List[float]] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Remember a new person or update existing.

        Args:
            name: Person's name
            face_embedding: 512-d face vector from DeepFace
            voice_embedding: 192-d voice vector from WeSpeaker
            notes: Initial notes about them

        Returns:
            True if successful
        """
        content = f"Met {name}"
        if notes:
            content += f". {notes}"

        context = {"person": name}
        if face_embedding:
            context["face_embedding"] = face_embedding
        if voice_embedding:
            context["voice_embedding"] = voice_embedding

        memory_id = await self.store(content, context=context)
        return memory_id is not None

    # =========================================================================
    # Health & Status
    # =========================================================================

    async def health_check(self) -> bool:
        """Check if MemoRable is reachable.

        Returns:
            True if healthy
        """
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def get_status(self) -> Dict[str, Any]:
        """Get system status and metrics.

        Returns:
            Status dict with memory count, open loops, etc.
        """
        session = await self._get_session()

        try:
            async with session.get(
                f"{self.base_url}/dashboard/json",
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"error": f"Status {resp.status}"}
        except Exception as e:
            return {"error": str(e)}


# =============================================================================
# Singleton Pattern
# =============================================================================

_client: Optional[MemoRableClient] = None


def get_memorable_client(
    base_url: str = "http://localhost:3000",
    entity: str = "chloe",
) -> MemoRableClient:
    """Get the singleton MemoRable client.

    Args:
        base_url: MemoRable server URL
        entity: Identity entity name

    Returns:
        MemoRableClient instance
    """
    global _client
    if _client is None:
        _client = MemoRableClient(base_url=base_url, entity=entity)
    return _client


# =============================================================================
# Convenience Functions for johnny5.py
# =============================================================================

async def on_startup() -> List[Memory]:
    """Called when robot starts - reconstruct temporal context.

    Returns:
        Relevant memories for "waking up"
    """
    client = get_memorable_client()

    if not await client.health_check():
        print("MemoRable not available - running without long-term memory")
        return []

    # "Where was I? What was I doing?"
    memories = await client.whats_relevant()

    if memories:
        print(f"MemoRable: Recalled {len(memories)} relevant memories")
        print(f"  Last context: {memories[0].content[:60]}...")
    else:
        print("MemoRable: No recent context to reconstruct")

    return memories


async def on_person_recognized(name: str) -> Optional[Briefing]:
    """Called when face/voice identifies someone.

    Args:
        name: Recognized person's name

    Returns:
        Briefing for conversation
    """
    client = get_memorable_client()
    briefing = await client.get_briefing(name)

    if briefing:
        print(f"MemoRable: Got briefing on {name}")
        if briefing.you_owe_them:
            print(f"  You owe them: {briefing.you_owe_them}")

    return briefing


async def on_conversation_end(
    person: Optional[str],
    summary: str,
) -> Optional[str]:
    """Called when conversation ends - store the interaction.

    Args:
        person: Who was talked to (if known)
        summary: Brief summary of conversation

    Returns:
        Memory ID
    """
    client = get_memorable_client()

    context = {}
    if person:
        context["person"] = person

    return await client.store(summary, context=context)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("Testing MemoRable Client")
        print("=" * 50)

        client = get_memorable_client()

        # Health check
        healthy = await client.health_check()
        print(f"Health check: {'✓' if healthy else '✗'}")

        if not healthy:
            print("MemoRable not running. Start with: docker-compose up -d")
            await client.close()
            return

        # Store a memory
        print("\nStoring test memory...")
        memory_id = await client.store(
            "Had a great conversation with Alan about robot hiking trails",
            context={"location": "lab", "emotion": "excited"},
        )
        print(f"  Stored: {memory_id}")

        # Recall
        print("\nRecalling 'hiking'...")
        memories = await client.recall("hiking", limit=5)
        for m in memories:
            print(f"  [{m.salience:.0f}] {m.content[:50]}...")

        # Briefing
        print("\nGetting briefing on 'Alan'...")
        briefing = await client.get_briefing("Alan")
        if briefing:
            print(f"  Last interaction: {briefing.last_interaction}")
            print(f"  Recent topics: {briefing.recent_topics}")

        # What's relevant
        print("\nWhat's relevant now?")
        relevant = await client.whats_relevant()
        for m in relevant[:3]:
            print(f"  [{m.salience:.0f}] {m.content[:50]}...")

        await client.close()
        print("\nTest complete!")

    asyncio.run(test())

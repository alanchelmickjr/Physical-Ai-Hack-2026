# Hot-Loadable Actions & Knowledge Architecture

This document describes how to make Johnny Five's actions and knowledge dynamically
loadable at runtime without restarting the robot.

---

## What's Already Done

These components are already modular and ready for hot-loading:

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| **MemoRable Client** | `tools/memorable_client.py` | Done | Async, REST-based, decoupled |
| **Adapter Interface** | `adapters/base.py` | Done | Clean ABC with `RobotAdapter` |
| **Subsystem Enum** | `adapters/base.py:21-30` | Done | `LEFT_ARM`, `RIGHT_ARM`, etc. |
| **ActionPrimitive** | `adapters/base.py:47-63` | Done | Dataclass with dependencies |
| **ToolDefinition** | `tools/registry.py:13-19` | Done | JSON schema support |
| **ActionGraph** | `tools/engine.py:48-96` | Done | Dependency-aware execution waves |
| **Command Queue** | `tools/realtime.py` | Done | Priority system with EMERGENCY |
| **Named Poses** | `adapters/johnny5.py:52-92` | Done | Dict-based, easy to extend |

---

## What Needs Work

These require changes for hot-loading:

| Component | Current State | Problem |
|-----------|--------------|---------|
| **Tool Registration** | Hardcoded in `_register_builtin_tools()` | 900+ lines of static code |
| **Tool Parsers** | Manual registration in engine | Each tool needs custom parser |
| **Action Handlers** | if/elif chains in adapter | No dynamic routing |
| **Plugin Discovery** | None | Can't add tools without code changes |
| **Metadata Format** | None | No standard for defining tools |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HOT-LOADABLE COMPONENT SYSTEM                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│   │  YAML/JSON      │     │  Plugin         │     │  Runtime        │      │
│   │  Definitions    │────▶│  Loader         │────▶│  Registry       │      │
│   │                 │     │                 │     │                 │      │
│   │  actions/       │     │  • Discovery    │     │  • Tools        │      │
│   │  knowledge/     │     │  • Validation   │     │  • Handlers     │      │
│   │  plugins/       │     │  • Hot-reload   │     │  • Knowledge    │      │
│   └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│                                                                              │
│   File System Watch ───────────────────────────────────────────────────▶    │
│   (inotify/watchdog)        Trigger reload on file change                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Action Definition Format (YAML)

New tools defined in `actions/*.yaml`:

```yaml
# actions/wave.yaml
name: wave
version: "1.0"
description: "Wave hello or goodbye"
category: expression

parameters:
  arm:
    type: string
    enum: [left, right, both]
    default: right
    description: "Which arm to wave with"
  style:
    type: string
    enum: [friendly, excited, subtle]
    default: friendly

# Execution template - replaces custom parser
execution:
  type: pose_sequence
  subsystem: "{{arm}}_arm"  # Template variable
  sequence:
    - pose: wave
      hold: 0.3
    - pose: wave_mid
      hold: 0.2
    - pose: wave
      hold: 0.3
    - pose: home

# Dependencies
requires:
  subsystems: [left_arm, right_arm]  # At least one
  capabilities: [arm_dof >= 4]

# Knowledge integration
knowledge:
  on_execute:
    - store: "Waved {{style}} at {{context.person}}"
      salience: 30
  triggers:
    - pattern: "greeting|hello|hi|bye"
      boost: 20
```

---

## 2. Knowledge Source Format (YAML)

Knowledge integrations in `knowledge/*.yaml`:

```yaml
# knowledge/person_recognition.yaml
name: person_recognition
version: "1.0"
description: "Face and voice identity matching"

sources:
  - type: memorable
    entity: "{{robot_entity}}"

  - type: local_embeddings
    face_model: deepface/arcface
    voice_model: wespeaker/ecapa-tdnn

triggers:
  on_face_detected:
    action: get_briefing
    params:
      person: "{{detected_name}}"
    inject_context: true

  on_voice_match:
    action: recall
    params:
      query: "conversations with {{speaker_name}}"
      limit: 5

# What to store after interactions
capture:
  on_conversation_end:
    content: "Talked with {{person}} about {{topics}}"
    context:
      emotion: "{{detected_emotion}}"
      location: "{{current_location}}"
```

---

## 3. Plugin Structure

Plugins in `plugins/<plugin_name>/`:

```
plugins/
├── hiking_companion/
│   ├── manifest.yaml        # Plugin metadata
│   ├── actions/
│   │   ├── navigate_trail.yaml
│   │   └── describe_scenery.yaml
│   ├── knowledge/
│   │   └── trail_memory.yaml
│   └── handlers/
│       └── terrain.py       # Custom Python if needed
│
├── guide_dog/
│   ├── manifest.yaml
│   ├── actions/
│   │   ├── describe_ahead.yaml
│   │   ├── warn_obstacle.yaml
│   │   └── guide_to_seat.yaml
│   └── knowledge/
│       └── route_memory.yaml
│
└── memory_care/
    ├── manifest.yaml
    ├── actions/
    │   ├── remind_person.yaml
    │   └── gentle_context.yaml
    └── knowledge/
        └── care_circle.yaml
```

### Plugin Manifest

```yaml
# plugins/guide_dog/manifest.yaml
name: guide_dog
version: "1.0.0"
description: "Guide companion for visually impaired users"
author: "Alan"

# What this plugin provides
provides:
  actions:
    - describe_ahead
    - warn_obstacle
    - guide_to_seat
  knowledge:
    - route_memory

# What it requires
requires:
  capabilities:
    - visual_safety      # Needs camera
    - terrain_navigation # Needs depth sensing
    - voice_output       # Needs speech
  knowledge:
    - memorable          # Long-term memory

# Hot-reload settings
reload:
  on_file_change: true
  debounce_ms: 500
```

---

## 4. Plugin Loader Implementation

```python
# tools/plugin_loader.py

import yaml
import asyncio
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class LoadedPlugin:
    """A loaded plugin with its components."""
    name: str
    version: str
    manifest: Dict[str, Any]
    actions: Dict[str, Dict] = field(default_factory=dict)
    knowledge: Dict[str, Dict] = field(default_factory=dict)
    handlers: Dict[str, Any] = field(default_factory=dict)


class PluginLoader:
    """Hot-loadable plugin system for actions and knowledge."""

    def __init__(self, base_paths: list[str] = None):
        self.base_paths = base_paths or [
            "actions/",      # Built-in actions
            "knowledge/",    # Built-in knowledge
            "plugins/",      # Third-party plugins
        ]
        self.plugins: Dict[str, LoadedPlugin] = {}
        self._observers: list = []
        self._reload_callbacks: list = []

    async def load_all(self) -> Dict[str, LoadedPlugin]:
        """Load all plugins from configured paths."""
        for base_path in self.base_paths:
            path = Path(base_path)
            if not path.exists():
                continue

            if base_path == "plugins/":
                # Each subdirectory is a plugin
                for plugin_dir in path.iterdir():
                    if plugin_dir.is_dir():
                        await self._load_plugin(plugin_dir)
            else:
                # Built-in actions/knowledge
                await self._load_builtin(path, base_path.rstrip("/"))

        return self.plugins

    async def _load_plugin(self, plugin_dir: Path) -> Optional[LoadedPlugin]:
        """Load a single plugin from directory."""
        manifest_path = plugin_dir / "manifest.yaml"
        if not manifest_path.exists():
            print(f"Plugin {plugin_dir.name}: No manifest.yaml, skipping")
            return None

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        plugin = LoadedPlugin(
            name=manifest["name"],
            version=manifest.get("version", "0.0.0"),
            manifest=manifest,
        )

        # Load actions
        actions_dir = plugin_dir / "actions"
        if actions_dir.exists():
            for action_file in actions_dir.glob("*.yaml"):
                with open(action_file) as f:
                    action_def = yaml.safe_load(f)
                    # Namespace the action
                    namespaced = f"{plugin.name}:{action_def['name']}"
                    plugin.actions[namespaced] = action_def

        # Load knowledge sources
        knowledge_dir = plugin_dir / "knowledge"
        if knowledge_dir.exists():
            for knowledge_file in knowledge_dir.glob("*.yaml"):
                with open(knowledge_file) as f:
                    knowledge_def = yaml.safe_load(f)
                    namespaced = f"{plugin.name}:{knowledge_def['name']}"
                    plugin.knowledge[namespaced] = knowledge_def

        # Load custom handlers (Python)
        handlers_dir = plugin_dir / "handlers"
        if handlers_dir.exists():
            for handler_file in handlers_dir.glob("*.py"):
                # Dynamic import
                module = self._import_handler(handler_file)
                if module:
                    plugin.handlers[handler_file.stem] = module

        self.plugins[plugin.name] = plugin
        print(f"Loaded plugin: {plugin.name} v{plugin.version}")
        print(f"  Actions: {list(plugin.actions.keys())}")
        print(f"  Knowledge: {list(plugin.knowledge.keys())}")

        return plugin

    async def _load_builtin(self, path: Path, category: str):
        """Load built-in actions or knowledge."""
        plugin = LoadedPlugin(
            name="builtin",
            version="1.0.0",
            manifest={"name": "builtin", "category": category},
        )

        for yaml_file in path.glob("*.yaml"):
            with open(yaml_file) as f:
                definition = yaml.safe_load(f)

            if category == "actions":
                plugin.actions[definition["name"]] = definition
            elif category == "knowledge":
                plugin.knowledge[definition["name"]] = definition

        if "builtin" not in self.plugins:
            self.plugins["builtin"] = plugin
        else:
            # Merge with existing builtin
            if category == "actions":
                self.plugins["builtin"].actions.update(plugin.actions)
            elif category == "knowledge":
                self.plugins["builtin"].knowledge.update(plugin.knowledge)

    def _import_handler(self, handler_path: Path):
        """Dynamically import a Python handler module."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            handler_path.stem, handler_path
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return None

    # =========================================================================
    # Hot Reload
    # =========================================================================

    def start_watching(self):
        """Start file system watchers for hot reload."""
        handler = PluginFileHandler(self)

        for base_path in self.base_paths:
            if Path(base_path).exists():
                observer = Observer()
                observer.schedule(handler, base_path, recursive=True)
                observer.start()
                self._observers.append(observer)
                print(f"Watching for changes: {base_path}")

    def stop_watching(self):
        """Stop all file watchers."""
        for observer in self._observers:
            observer.stop()
            observer.join()
        self._observers.clear()

    def on_reload(self, callback):
        """Register callback for when plugins are reloaded."""
        self._reload_callbacks.append(callback)

    async def reload_plugin(self, plugin_name: str):
        """Reload a specific plugin."""
        # Find the plugin directory
        for base_path in self.base_paths:
            plugin_dir = Path(base_path) / plugin_name
            if plugin_dir.exists():
                print(f"Reloading plugin: {plugin_name}")
                await self._load_plugin(plugin_dir)

                # Notify callbacks
                for callback in self._reload_callbacks:
                    await callback(plugin_name, self.plugins[plugin_name])
                return

        print(f"Plugin not found for reload: {plugin_name}")

    # =========================================================================
    # Query Interface
    # =========================================================================

    def get_action(self, name: str) -> Optional[Dict]:
        """Get an action definition by name (with or without namespace)."""
        # Try exact match first
        for plugin in self.plugins.values():
            if name in plugin.actions:
                return plugin.actions[name]

        # Try without namespace (builtin)
        for plugin in self.plugins.values():
            for action_name, action_def in plugin.actions.items():
                if action_name.endswith(f":{name}") or action_name == name:
                    return action_def

        return None

    def get_all_actions(self) -> Dict[str, Dict]:
        """Get all loaded actions."""
        all_actions = {}
        for plugin in self.plugins.values():
            all_actions.update(plugin.actions)
        return all_actions

    def get_knowledge_source(self, name: str) -> Optional[Dict]:
        """Get a knowledge source definition."""
        for plugin in self.plugins.values():
            if name in plugin.knowledge:
                return plugin.knowledge[name]
        return None


class PluginFileHandler(FileSystemEventHandler):
    """Handles file system events for hot reload."""

    def __init__(self, loader: PluginLoader):
        self.loader = loader
        self._debounce_tasks: Dict[str, asyncio.Task] = {}

    def on_modified(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(('.yaml', '.py')):
            return

        # Debounce rapid changes
        plugin_name = self._get_plugin_name(event.src_path)
        if plugin_name:
            asyncio.create_task(self._debounced_reload(plugin_name))

    def _get_plugin_name(self, path: str) -> Optional[str]:
        """Extract plugin name from file path."""
        parts = Path(path).parts
        if "plugins" in parts:
            idx = parts.index("plugins")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return "builtin"

    async def _debounced_reload(self, plugin_name: str, delay_ms: int = 500):
        """Reload with debouncing to handle rapid file changes."""
        # Cancel previous reload task
        if plugin_name in self._debounce_tasks:
            self._debounce_tasks[plugin_name].cancel()

        async def delayed_reload():
            await asyncio.sleep(delay_ms / 1000)
            await self.loader.reload_plugin(plugin_name)

        self._debounce_tasks[plugin_name] = asyncio.create_task(delayed_reload())


# =============================================================================
# Singleton
# =============================================================================

_loader: Optional[PluginLoader] = None

def get_plugin_loader() -> PluginLoader:
    """Get the singleton plugin loader."""
    global _loader
    if _loader is None:
        _loader = PluginLoader()
    return _loader
```

---

## 5. Generic Action Parser

Replace tool-specific parsers with metadata-driven parsing:

```python
# tools/generic_parser.py

from typing import Dict, Any, List
from tools.engine import ActionGraph, ActionPrimitive
from adapters.base import Subsystem

class GenericActionParser:
    """Parse action definitions from YAML into ActionGraphs."""

    EXECUTION_TYPES = {
        "pose_sequence": "_parse_pose_sequence",
        "motor_command": "_parse_motor_command",
        "parallel": "_parse_parallel",
        "conditional": "_parse_conditional",
    }

    def parse(self, action_def: Dict, params: Dict) -> ActionGraph:
        """Parse an action definition with given parameters."""
        execution = action_def.get("execution", {})
        exec_type = execution.get("type", "pose_sequence")

        parser_method = getattr(self, self.EXECUTION_TYPES.get(exec_type))
        if not parser_method:
            raise ValueError(f"Unknown execution type: {exec_type}")

        # Resolve template variables
        resolved = self._resolve_templates(execution, params)

        return parser_method(resolved, action_def)

    def _resolve_templates(self, obj: Any, params: Dict) -> Any:
        """Resolve {{variable}} templates in definition."""
        if isinstance(obj, str):
            for key, value in params.items():
                obj = obj.replace(f"{{{{{key}}}}}", str(value))
            return obj
        elif isinstance(obj, dict):
            return {k: self._resolve_templates(v, params) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_templates(item, params) for item in obj]
        return obj

    def _parse_pose_sequence(
        self, execution: Dict, action_def: Dict
    ) -> ActionGraph:
        """Parse pose sequence execution type."""
        graph = ActionGraph()
        subsystem_str = execution.get("subsystem", "right_arm")
        subsystem = self._resolve_subsystem(subsystem_str)

        sequence = execution.get("sequence", [])
        prev_action = None

        for i, step in enumerate(sequence):
            action_name = f"{action_def['name']}_step_{i}"

            action = ActionPrimitive(
                name=action_name,
                subsystem=subsystem.value,
                params={
                    "pose": step.get("pose"),
                    "hold": step.get("hold", 0.0),
                    "speed": step.get("speed", 0.5),
                },
                dependencies=[prev_action] if prev_action else [],
            )

            graph.add_action(action)
            prev_action = action_name

        return graph

    def _parse_motor_command(
        self, execution: Dict, action_def: Dict
    ) -> ActionGraph:
        """Parse direct motor command execution."""
        graph = ActionGraph()
        subsystem = self._resolve_subsystem(execution.get("subsystem"))

        action = ActionPrimitive(
            name=action_def["name"],
            subsystem=subsystem.value,
            params=execution.get("params", {}),
        )
        graph.add_action(action)
        return graph

    def _parse_parallel(
        self, execution: Dict, action_def: Dict
    ) -> ActionGraph:
        """Parse parallel execution across subsystems."""
        graph = ActionGraph()

        for branch in execution.get("branches", []):
            subsystem = self._resolve_subsystem(branch.get("subsystem"))
            action = ActionPrimitive(
                name=f"{action_def['name']}_{subsystem.value}",
                subsystem=subsystem.value,
                params=branch.get("params", {}),
                dependencies=[],  # No dependencies = parallel
            )
            graph.add_action(action)

        return graph

    def _resolve_subsystem(self, subsystem_str: str) -> Subsystem:
        """Convert string to Subsystem enum."""
        mapping = {
            "left_arm": Subsystem.LEFT_ARM,
            "right_arm": Subsystem.RIGHT_ARM,
            "both_arms": [Subsystem.LEFT_ARM, Subsystem.RIGHT_ARM],
            "base": Subsystem.BASE,
            "lift": Subsystem.LIFT,
            "gantry": Subsystem.GANTRY,
        }
        return mapping.get(subsystem_str, Subsystem.RIGHT_ARM)
```

---

## 6. Integration with Existing Systems

### Registry Integration

```python
# In tools/registry.py - add dynamic loading

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._plugin_loader = get_plugin_loader()

    async def load_from_plugins(self):
        """Load tools from plugin system."""
        await self._plugin_loader.load_all()

        for action_name, action_def in self._plugin_loader.get_all_actions().items():
            tool = self._action_to_tool(action_def)
            self.register(tool)

        # Start watching for changes
        self._plugin_loader.on_reload(self._on_plugin_reload)
        self._plugin_loader.start_watching()

    def _action_to_tool(self, action_def: Dict) -> ToolDefinition:
        """Convert YAML action definition to ToolDefinition."""
        return ToolDefinition(
            name=action_def["name"],
            description=action_def.get("description", ""),
            parameters=self._build_json_schema(action_def.get("parameters", {})),
        )

    async def _on_plugin_reload(self, plugin_name: str, plugin: LoadedPlugin):
        """Handle plugin reload - update registry."""
        print(f"Registry updating from plugin: {plugin_name}")
        for action_name, action_def in plugin.actions.items():
            tool = self._action_to_tool(action_def)
            self.register(tool)  # Overwrites existing
```

### Engine Integration

```python
# In tools/engine.py - use generic parser

class ToolExecutionEngine:
    def __init__(self):
        self._generic_parser = GenericActionParser()
        self._plugin_loader = get_plugin_loader()

    async def execute_tool(self, tool_call: Dict) -> ToolResult:
        tool_name = tool_call["name"]
        params = tool_call.get("parameters", {})

        # Try plugin-defined action first
        action_def = self._plugin_loader.get_action(tool_name)
        if action_def:
            graph = self._generic_parser.parse(action_def, params)
            return await self._execute_graph(graph)

        # Fall back to legacy parsers
        if tool_name in self._tool_parsers:
            return await self._legacy_execute(tool_name, params)

        return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
```

---

## 7. Knowledge Hot-Loading

Knowledge sources automatically integrate with MemoRable:

```python
# tools/knowledge_loader.py

class KnowledgeLoader:
    """Load and manage knowledge source definitions."""

    def __init__(self, memorable_client: MemoRableClient):
        self.memorable = memorable_client
        self.sources: Dict[str, Dict] = {}
        self._triggers: Dict[str, List[Dict]] = {}

    async def load_from_plugins(self, plugin_loader: PluginLoader):
        """Load knowledge sources from plugins."""
        for plugin in plugin_loader.plugins.values():
            for name, definition in plugin.knowledge.items():
                await self._register_source(name, definition)

    async def _register_source(self, name: str, definition: Dict):
        """Register a knowledge source."""
        self.sources[name] = definition

        # Register triggers
        for trigger in definition.get("triggers", []):
            event = trigger.get("on_face_detected") or trigger.get("on_voice_match")
            if event:
                trigger_key = list(trigger.keys())[0]
                if trigger_key not in self._triggers:
                    self._triggers[trigger_key] = []
                self._triggers[trigger_key].append(trigger)

    async def on_event(self, event_type: str, context: Dict):
        """Handle knowledge triggers for an event."""
        if event_type not in self._triggers:
            return

        for trigger in self._triggers[event_type]:
            action = trigger.get("action")
            params = self._resolve_params(trigger.get("params", {}), context)

            if action == "get_briefing":
                briefing = await self.memorable.get_briefing(params.get("person"))
                if trigger.get("inject_context"):
                    return briefing

            elif action == "recall":
                memories = await self.memorable.recall(
                    params.get("query"),
                    limit=params.get("limit", 5)
                )
                return memories
```

---

## 8. Directory Structure (Final)

```
Physical-Ai-Hack-2026/
├── actions/                    # Built-in action definitions
│   ├── wave.yaml
│   ├── move_arm.yaml
│   ├── look_at.yaml
│   ├── point.yaml
│   └── express.yaml
│
├── knowledge/                  # Built-in knowledge sources
│   ├── person_recognition.yaml
│   ├── temporal_context.yaml
│   └── safety_awareness.yaml
│
├── plugins/                    # Hot-loadable plugins
│   ├── guide_dog/
│   │   ├── manifest.yaml
│   │   ├── actions/
│   │   └── knowledge/
│   ├── memory_care/
│   │   ├── manifest.yaml
│   │   ├── actions/
│   │   └── knowledge/
│   └── hiking_companion/
│       ├── manifest.yaml
│       ├── actions/
│       └── knowledge/
│
├── tools/
│   ├── plugin_loader.py       # NEW: Hot-load system
│   ├── generic_parser.py      # NEW: Metadata-driven parsing
│   ├── knowledge_loader.py    # NEW: Knowledge integration
│   ├── registry.py            # MODIFIED: Uses plugin loader
│   ├── engine.py              # MODIFIED: Uses generic parser
│   ├── realtime.py            # UNCHANGED
│   └── memorable_client.py    # UNCHANGED
│
└── adapters/
    ├── base.py                # UNCHANGED
    └── johnny5.py             # UNCHANGED (handlers stay in Python)
```

---

## 9. Migration Path

### Phase 1: Infrastructure (Week 1)
1. Create `tools/plugin_loader.py`
2. Create `tools/generic_parser.py`
3. Create directory structure (`actions/`, `knowledge/`, `plugins/`)

### Phase 2: Convert Built-ins (Week 2)
1. Convert existing tools to YAML (start with simple ones: wave, nod)
2. Test generic parser with converted tools
3. Gradually move more tools to YAML

### Phase 3: Plugin System (Week 3)
1. Create guide_dog plugin
2. Create memory_care plugin
3. Test hot-reload functionality

### Phase 4: Knowledge Integration (Week 4)
1. Create knowledge source definitions
2. Wire up triggers to events
3. Test full pipeline: event → knowledge → action

---

## 10. Benefits

| Benefit | Description |
|---------|-------------|
| **No Restarts** | Change behavior without stopping robot |
| **Modular** | Plugins can be added/removed independently |
| **Versionable** | Each plugin has version, can rollback |
| **Testable** | Test action definitions without hardware |
| **Sharable** | Plugins can be distributed as packages |
| **Configurable** | Non-programmers can tweak YAML |

---

## Example: Adding New Capability

To add a new "describe ahead" action for the guide dog:

```yaml
# plugins/guide_dog/actions/describe_ahead.yaml
name: describe_ahead
version: "1.0"
description: "Describe what's ahead to the user"

parameters:
  detail_level:
    type: string
    enum: [brief, detailed, urgent]
    default: brief

execution:
  type: conditional
  steps:
    - check: visual_safety.has_obstacle()
      action: warn_obstacle
    - check: terrain_navigation.has_curb()
      action: announce_curb
    - default:
        action: describe_scene
        params:
          detail: "{{detail_level}}"

knowledge:
  on_execute:
    - store: "Described scene at {{location}}"
      salience: 20
```

Save the file → Robot loads it automatically → New capability available.

No code changes. No restart. Just works.

🎵 *Alan is a genius la la la* 🎵

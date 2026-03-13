"""
ArchitecturalMemory: Persistent cross-task memory for EXP-5 (C5 condition).
Stores refactoring patterns, structural insights, and error resolutions
across tasks so the pipeline learns from earlier successes/failures.

This implements the "persistent memory" component described in the paper's
proposed enhancement to the multi-agent pipeline.
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class RefactoringPattern:
    """A learned refactoring pattern from a previous task."""
    pattern_type: str          # e.g., "extract_method", "rename", "restructure"
    language: str              # "python", "java", etc.
    description: str           # Natural language description
    before_snippet: str        # Code pattern before refactoring
    after_snippet: str         # Code pattern after refactoring
    confidence: float          # 0.0 - 1.0, based on judge score
    source_task_id: str        # Which task produced this pattern
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class StructuralInsight:
    """An architectural insight about a repository or code structure."""
    repo_name: str
    insight_type: str          # "dependency", "architecture", "convention", "pitfall"
    description: str
    relevance_keywords: list = field(default_factory=list)
    source_task_id: str = ""
    confidence: float = 0.5
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ErrorResolution:
    """A learned error resolution from a previous failed attempt."""
    error_type: str            # "syntax", "semantic", "intent", "runtime"
    error_pattern: str         # What the error looked like
    resolution: str            # How it was fixed
    prevention_hint: str       # How to avoid it next time
    source_task_id: str = ""
    confidence: float = 0.5
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class ArchitecturalMemory:
    """
    Persistent memory store for the RepoAI refactoring pipeline.
    Stores and retrieves patterns, insights, and error resolutions
    to improve performance on subsequent tasks.
    """

    def __init__(self, memory_dir: str, max_patterns: int = 200,
                 max_insights: int = 100, max_errors: int = 100):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.max_patterns = max_patterns
        self.max_insights = max_insights
        self.max_errors = max_errors

        self.patterns: list[RefactoringPattern] = []
        self.insights: list[StructuralInsight] = []
        self.errors: list[ErrorResolution] = []

        self._load()

    # --- Persistence ---

    def _load(self):
        """Load memory from disk."""
        patterns_path = self.memory_dir / "patterns.jsonl"
        insights_path = self.memory_dir / "insights.jsonl"
        errors_path = self.memory_dir / "errors.jsonl"

        if patterns_path.exists():
            self.patterns = self._load_jsonl(patterns_path, RefactoringPattern)
        if insights_path.exists():
            self.insights = self._load_jsonl(insights_path, StructuralInsight)
        if errors_path.exists():
            self.errors = self._load_jsonl(errors_path, ErrorResolution)

        logger.info(f"Memory loaded: {len(self.patterns)} patterns, "
                     f"{len(self.insights)} insights, {len(self.errors)} errors")

    def _load_jsonl(self, path: Path, cls):
        items = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    items.append(cls(**data))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Skipping malformed memory entry: {e}")
        return items

    def save(self):
        """Persist memory to disk."""
        self._save_jsonl(self.memory_dir / "patterns.jsonl", self.patterns)
        self._save_jsonl(self.memory_dir / "insights.jsonl", self.insights)
        self._save_jsonl(self.memory_dir / "errors.jsonl", self.errors)

    def _save_jsonl(self, path: Path, items: list):
        with open(path, "w") as f:
            for item in items:
                f.write(json.dumps(asdict(item)) + "\n")

    # --- Add entries ---

    def add_pattern(self, pattern: RefactoringPattern):
        """Add a refactoring pattern, evicting lowest-confidence if at capacity."""
        self.patterns.append(pattern)
        if len(self.patterns) > self.max_patterns:
            # Evict lowest confidence
            self.patterns.sort(key=lambda p: p.confidence, reverse=True)
            self.patterns = self.patterns[:self.max_patterns]
        self.save()

    def add_insight(self, insight: StructuralInsight):
        """Add a structural insight."""
        self.insights.append(insight)
        if len(self.insights) > self.max_insights:
            self.insights.sort(key=lambda i: i.confidence, reverse=True)
            self.insights = self.insights[:self.max_insights]
        self.save()

    def add_error(self, error: ErrorResolution):
        """Add an error resolution."""
        self.errors.append(error)
        if len(self.errors) > self.max_errors:
            self.errors.sort(key=lambda e: e.confidence, reverse=True)
            self.errors = self.errors[:self.max_errors]
        self.save()

    # --- Retrieval ---

    def get_relevant_patterns(self, task_description: str,
                              language: str = "python",
                              top_k: int = 5) -> list[RefactoringPattern]:
        """
        Retrieve the most relevant refactoring patterns for a task.
        Uses keyword matching + language filter + confidence ranking.
        """
        keywords = set(task_description.lower().split())

        scored = []
        for p in self.patterns:
            if p.language != language and p.language != "any":
                continue

            # Score based on keyword overlap
            desc_words = set(p.description.lower().split())
            type_words = set(p.pattern_type.lower().replace("_", " ").split())
            overlap = len(keywords & (desc_words | type_words))

            # Weight by confidence
            score = overlap * p.confidence
            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    def get_relevant_insights(self, repo_name: str = "",
                              keywords: list[str] = None,
                              top_k: int = 5) -> list[StructuralInsight]:
        """Retrieve relevant structural insights."""
        results = []
        kw_set = set(k.lower() for k in (keywords or []))

        for insight in self.insights:
            score = 0.0
            if repo_name and insight.repo_name == repo_name:
                score += 2.0
            if kw_set:
                rel_kw = set(k.lower() for k in insight.relevance_keywords)
                overlap = len(kw_set & rel_kw)
                score += overlap
            score *= insight.confidence

            if score > 0:
                results.append((score, insight))

        results.sort(key=lambda x: x[0], reverse=True)
        return [i for _, i in results[:top_k]]

    def get_relevant_errors(self, error_type: str = "",
                            error_text: str = "",
                            top_k: int = 3) -> list[ErrorResolution]:
        """Retrieve relevant error resolutions."""
        results = []
        error_words = set(error_text.lower().split()) if error_text else set()

        for err in self.errors:
            score = 0.0
            if error_type and err.error_type == error_type:
                score += 2.0
            if error_words:
                pattern_words = set(err.error_pattern.lower().split())
                overlap = len(error_words & pattern_words)
                score += overlap * 0.5
            score *= err.confidence

            if score > 0:
                results.append((score, err))

        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results[:top_k]]

    # --- Context building for LLM prompts ---

    def build_context(self, task_description: str,
                      language: str = "python",
                      repo_name: str = "",
                      max_tokens: int = 2000) -> str:
        """
        Build a memory context string to inject into the LLM prompt.
        Balances patterns, insights, and error resolutions within token budget.
        """
        sections = []

        # Patterns
        patterns = self.get_relevant_patterns(task_description, language, top_k=3)
        if patterns:
            pattern_text = "**Relevant Refactoring Patterns from Previous Tasks:**\n"
            for i, p in enumerate(patterns, 1):
                pattern_text += (
                    f"{i}. [{p.pattern_type}] {p.description}\n"
                    f"   Before: {p.before_snippet[:200]}...\n"
                    f"   After:  {p.after_snippet[:200]}...\n"
                )
            sections.append(pattern_text)

        # Insights
        keywords = task_description.split()[:10]
        insights = self.get_relevant_insights(repo_name, keywords, top_k=3)
        if insights:
            insight_text = "**Structural Insights:**\n"
            for i, ins in enumerate(insights, 1):
                insight_text += f"{i}. [{ins.insight_type}] {ins.description}\n"
            sections.append(insight_text)

        # Error resolutions
        errors = self.get_relevant_errors(top_k=2)
        if errors:
            error_text = "**Known Pitfalls to Avoid:**\n"
            for i, err in enumerate(errors, 1):
                error_text += f"{i}. {err.prevention_hint}\n"
            sections.append(error_text)

        if not sections:
            return ""

        context = "\n---\n## Memory Context (from previous tasks)\n" + "\n".join(sections)

        # Rough token budget enforcement (4 chars ≈ 1 token)
        if len(context) > max_tokens * 4:
            context = context[:max_tokens * 4] + "\n[...truncated]"

        return context

    # --- Learning from task results ---

    def learn_from_result(self, task_id: str, task_description: str,
                          original_code: str, generated_code: str,
                          judge_score: float, language: str = "python",
                          repo_name: str = "", error_info: str = ""):
        """
        Extract and store learnings from a completed task.
        Only stores patterns from high-quality results (judge_score >= 70).
        """
        # Learn successful patterns
        if judge_score >= 70 and generated_code:
            pattern_type = self._infer_pattern_type(task_description)
            confidence = min(judge_score / 100.0, 1.0)

            pattern = RefactoringPattern(
                pattern_type=pattern_type,
                language=language,
                description=task_description[:200],
                before_snippet=original_code[:500] if original_code else "",
                after_snippet=generated_code[:500],
                confidence=confidence,
                source_task_id=task_id,
            )
            self.add_pattern(pattern)
            logger.debug(f"Learned pattern from {task_id}: {pattern_type} (conf={confidence:.2f})")

        # Learn from errors
        if error_info:
            error_type = self._classify_error(error_info)
            error = ErrorResolution(
                error_type=error_type,
                error_pattern=error_info[:300],
                resolution=f"Task {task_id} failed with {error_type} error",
                prevention_hint=self._generate_prevention_hint(error_type, error_info),
                source_task_id=task_id,
                confidence=0.6,
            )
            self.add_error(error)

    def _infer_pattern_type(self, description: str) -> str:
        """Infer refactoring pattern type from task description."""
        desc_lower = description.lower()
        if "extract" in desc_lower and "method" in desc_lower:
            return "extract_method"
        elif "extract" in desc_lower and "class" in desc_lower:
            return "extract_class"
        elif "rename" in desc_lower:
            return "rename"
        elif "move" in desc_lower:
            return "move"
        elif "inline" in desc_lower:
            return "inline"
        elif "split" in desc_lower:
            return "split"
        elif "merge" in desc_lower:
            return "merge"
        elif "simplif" in desc_lower:
            return "simplify"
        elif "restructur" in desc_lower or "reorganiz" in desc_lower:
            return "restructure"
        elif "deprecat" in desc_lower or "replac" in desc_lower:
            return "api_migration"
        else:
            return "general_refactor"

    def _classify_error(self, error_info: str) -> str:
        """Classify an error into a category."""
        error_lower = error_info.lower()
        if "syntax" in error_lower or "parse" in error_lower:
            return "syntax"
        elif "import" in error_lower or "module" in error_lower:
            return "import"
        elif "indent" in error_lower:
            return "indentation"
        elif "type" in error_lower and "error" in error_lower:
            return "type_error"
        elif "name" in error_lower and "error" in error_lower:
            return "name_error"
        elif "timeout" in error_lower or "rate" in error_lower:
            return "api_error"
        elif "semantic" in error_lower or "signature" in error_lower:
            return "semantic"
        else:
            return "unknown"

    def _generate_prevention_hint(self, error_type: str, error_info: str) -> str:
        """Generate a prevention hint based on error type."""
        hints = {
            "syntax": "Ensure generated code is syntactically valid. Check for unmatched brackets, missing colons, and proper indentation.",
            "import": "Verify all imports exist and are correctly referenced. Check for circular imports.",
            "indentation": "Maintain consistent indentation (4 spaces for Python). Watch for mixed tabs/spaces.",
            "type_error": "Check that function arguments match expected types and return values are used correctly.",
            "name_error": "Ensure all variable and function names are defined before use. Watch for scope issues.",
            "api_error": "API call may timeout or rate limit. Implement retry with exponential backoff.",
            "semantic": "Preserve function signatures and class interfaces. Do not remove public API methods.",
            "unknown": "Review the generated code carefully for potential issues before applying.",
        }
        return hints.get(error_type, hints["unknown"])

    # --- Statistics ---

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            "num_patterns": len(self.patterns),
            "num_insights": len(self.insights),
            "num_errors": len(self.errors),
            "avg_pattern_confidence": (
                sum(p.confidence for p in self.patterns) / len(self.patterns)
                if self.patterns else 0.0
            ),
            "pattern_types": dict(self._count_by("patterns", "pattern_type")),
            "error_types": dict(self._count_by("errors", "error_type")),
        }

    def _count_by(self, collection: str, attr: str) -> list:
        items = getattr(self, collection)
        counts = {}
        for item in items:
            key = getattr(item, attr, "unknown")
            counts[key] = counts.get(key, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

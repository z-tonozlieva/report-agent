# entity_config.py
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TeamConfig:
    """Configuration for a team"""

    name: str
    aliases: List[str]
    roles: List[str]
    description: str


@dataclass
class TechnologyConfig:
    """Configuration for a technology"""

    name: str
    aliases: List[str]
    category: str


@dataclass
class ProjectConfig:
    """Configuration for a project type"""

    name: str
    aliases: List[str]


@dataclass
class FocusAreaConfig:
    """Configuration for focus areas"""

    name: str
    aliases: List[str]
    category: str
    priority: Optional[str] = None


class EntityConfigLoader:
    """Loads and manages entity configurations from YAML files"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config/entities.yaml in the backend directory
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(backend_dir, "config", "entities.yaml")

        self.config_path = config_path
        self.teams: List[TeamConfig] = []
        self.technologies: List[TechnologyConfig] = []
        self.projects: List[ProjectConfig] = []
        self.focus_areas: List[FocusAreaConfig] = []
        self.temporal_patterns: Dict = {}
        self.organization: Dict = {}

        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(
                    f"Entity config file not found at {self.config_path}, using defaults"
                )
                self._load_defaults()
                return

            with open(self.config_path, encoding="utf-8") as file:
                config = yaml.safe_load(file)

            # Load teams
            self.teams = [
                TeamConfig(
                    name=team["name"],
                    aliases=team.get("aliases", []),
                    roles=team.get("roles", []),
                    description=team.get("description", ""),
                )
                for team in config.get("teams", [])
            ]

            # Load technologies
            self.technologies = [
                TechnologyConfig(
                    name=tech["name"],
                    aliases=tech.get("aliases", []),
                    category=tech.get("category", "general"),
                )
                for tech in config.get("technologies", [])
            ]

            # Load projects
            self.projects = [
                ProjectConfig(name=proj["name"], aliases=proj.get("aliases", []))
                for proj in config.get("projects", [])
            ]

            # Load focus areas
            self.focus_areas = [
                FocusAreaConfig(
                    name=area["name"],
                    aliases=area.get("aliases", []),
                    category=area.get("category", "general"),
                    priority=area.get("priority"),
                )
                for area in config.get("focus_areas", [])
            ]

            # Load temporal patterns and organization info
            self.temporal_patterns = config.get("temporal_patterns", {})
            self.organization = config.get("organization", {})

            logger.info(
                f"Loaded entity config: {len(self.teams)} teams, {len(self.technologies)} technologies"
            )

        except Exception as e:
            logger.error(f"Error loading entity config: {str(e)}")
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load minimal default configuration if file is not available"""
        self.teams = [
            TeamConfig(
                "Frontend Team",
                ["frontend", "ui"],
                ["Frontend Developer"],
                "UI development",
            ),
            TeamConfig(
                "Backend Team",
                ["backend", "api"],
                ["Backend Developer"],
                "Server development",
            ),
            TeamConfig(
                "DevOps Team", ["devops", "ops"], ["DevOps Engineer"], "Infrastructure"
            ),
        ]

        self.technologies = [
            TechnologyConfig("JavaScript", ["javascript", "js"], "frontend"),
            TechnologyConfig("Python", ["python"], "backend"),
            TechnologyConfig("API", ["api", "rest"], "integration"),
        ]

        self.focus_areas = [
            FocusAreaConfig("Bug", ["bug", "issue", "error"], "issue"),
            FocusAreaConfig("Achievement", ["completed", "success"], "achievement"),
        ]

    def get_team_patterns(self) -> List[str]:
        """Generate regex patterns for team detection"""
        patterns = []

        for team in self.teams:
            # Create pattern for team name and aliases
            all_terms = [team.name.lower()] + [alias.lower() for alias in team.aliases]

            # Escape special regex characters and create pattern
            escaped_terms = [re.escape(term) for term in all_terms]
            pattern = r"\b(" + "|".join(escaped_terms) + r")\b"
            patterns.append(pattern)

        return patterns

    def get_technology_patterns(self) -> List[str]:
        """Generate regex patterns for technology detection"""
        patterns = []

        for tech in self.technologies:
            all_terms = [tech.name.lower()] + [alias.lower() for alias in tech.aliases]
            escaped_terms = [re.escape(term) for term in all_terms]
            pattern = r"\b(" + "|".join(escaped_terms) + r")\b"
            patterns.append(pattern)

        return patterns

    def get_focus_area_patterns(self) -> List[str]:
        """Generate regex patterns for focus area detection"""
        patterns = []

        for area in self.focus_areas:
            all_terms = [area.name.lower()] + [alias.lower() for alias in area.aliases]
            escaped_terms = [re.escape(term) for term in all_terms]
            pattern = r"\b(" + "|".join(escaped_terms) + r")\b"
            patterns.append(pattern)

        return patterns

    def get_project_patterns(self) -> List[str]:
        """Generate regex patterns for project detection"""
        patterns = []

        for project in self.projects:
            all_terms = [project.name.lower()] + [
                alias.lower() for alias in project.aliases
            ]
            escaped_terms = [re.escape(term) for term in all_terms]
            pattern = r"\b(" + "|".join(escaped_terms) + r")\b"
            patterns.append(pattern)

        return patterns

    def find_matching_teams(self, query: str) -> Set[str]:
        """Find teams mentioned in a query"""
        query_lower = query.lower()
        matching_teams = set()

        for team in self.teams:
            # Check team name
            if team.name.lower() in query_lower:
                matching_teams.add(team.name)

            # Check aliases
            for alias in team.aliases:
                if alias.lower() in query_lower:
                    matching_teams.add(team.name)

        return matching_teams

    def find_matching_technologies(self, query: str) -> Set[str]:
        """Find technologies mentioned in a query"""
        query_lower = query.lower()
        matching_techs = set()

        for tech in self.technologies:
            # Check technology name
            if tech.name.lower() in query_lower:
                matching_techs.add(tech.name)

            # Check aliases
            for alias in tech.aliases:
                if alias.lower() in query_lower:
                    matching_techs.add(tech.name)

        return matching_techs

    def find_matching_focus_areas(self, query: str) -> Set[str]:
        """Find focus areas mentioned in a query"""
        query_lower = query.lower()
        matching_areas = set()

        for area in self.focus_areas:
            # Check area name
            if area.name.lower() in query_lower:
                matching_areas.add(area.name)

            # Check aliases
            for alias in area.aliases:
                if alias.lower() in query_lower:
                    matching_areas.add(area.name)

        return matching_areas

    def find_matching_projects(self, query: str) -> Set[str]:
        """Find projects mentioned in a query"""
        query_lower = query.lower()
        matching_projects = set()

        for project in self.projects:
            # Check project name
            if project.name.lower() in query_lower:
                matching_projects.add(project.name)

            # Check aliases
            for alias in project.aliases:
                if alias.lower() in query_lower:
                    matching_projects.add(project.name)

        return matching_projects

    def get_team_by_role(self, role: str) -> Optional[str]:
        """Find which team a role belongs to"""
        role_lower = role.lower()

        for team in self.teams:
            for team_role in team.roles:
                if team_role.lower() == role_lower:
                    return team.name

        return None

    def reload_config(self) -> None:
        """Reload configuration from file"""
        logger.info("Reloading entity configuration...")
        self._load_config()


# Global instance - will be initialized in main app
entity_config: Optional[EntityConfigLoader] = None


def get_entity_config() -> EntityConfigLoader:
    """Get the global entity configuration instance"""
    global entity_config
    if entity_config is None:
        entity_config = EntityConfigLoader()
    return entity_config


def initialize_entity_config(config_path: Optional[str] = None) -> EntityConfigLoader:
    """Initialize the global entity configuration"""
    global entity_config
    entity_config = EntityConfigLoader(config_path)
    return entity_config

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# tools
from oss.tools import list_doc_tool


@CrewBase
class LegalExpertCrew:
    """EU Legal Expert Crew"""

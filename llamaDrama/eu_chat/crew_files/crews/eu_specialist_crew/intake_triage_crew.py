# Crew Definition for my Initial Triage Agent
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from typing import List, Optional
from crewai_tools import SerperDevTool

from llamaDrama.eu_chat.oss.tools import list_doc_tool
# Dict structures


class AudienceProfile(BaseModel):
    technical_tier: str = Field(description="Technical OR Non-Technical")
    organizational_role: str = Field(
        description="Exact job category or profile identified")
    primary_compliance_focus: str = Field(
        description="The specific operational layer the user cares about")


class TriageOutputSchema(BaseModel):
    is_sufficiently_narrow: bool = Field(
        description="True if query is narrow and clear enough to process, False otherwise")
    clarification_question: Optional[str] = Field(
        None, description="The precise question or session reset error message")
    role_extracted: Optional[str] = Field(
        None, description="Supply chain identity: Provider, Deployer, or Exempt")
    jurisdiction_extracted: Optional[str] = Field(
        None, description="Geographic scope parameters mapping to Article 2")
    purpose_extracted: Optional[str] = Field(
        None, description="Predicted risk classification tier")
    audience_profile: Optional[AudienceProfile] = Field(
        None, description="The targeted user profile metadata")
    generated_subqueries: List[str] = Field(
        default_factory=list, description="Targeted vector database search strings")
    agent_1_assumptions: List[str] = Field(
        default_factory=list, description="Deductive hypotheses explaining the subqueries")


@CrewBase
class TriageCrew():
    """Crew for handling the Intake, Triage, and Search Strategy formulation"""

    # Points to your config folders where the YAMLs live
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def intake_triage_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['intake_triage_specialist'],
            tools=[list_doc_tool, SerperDevTool()],  # Only access this one
            verbose=True,
            allow_delegation=False
        )

    @task
    def triage_and_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['triage_and_strategy_task'],
            agent=self.intake_triage_specialist(),
            output_json=TriageOutputSchema  # enforce schema
        )

    @crew
    def crew(self) -> Crew:
        """Creates the isolated Triage Crew"""
        return Crew(
            agents=self.agents,  # get all agents
            tasks=self.tasks,   # get all tasks
            process=Process.sequential,  # do i really have a choice?
            tracing=True,
            verbose=True
        )

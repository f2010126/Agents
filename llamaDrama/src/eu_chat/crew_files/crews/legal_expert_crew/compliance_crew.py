# Crew Definition for my Enforcer and Compliance Agent
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from typing import List
from crewai.agents.agent_builder.base_agent import BaseAgent
# tools
from llamaDrama.eu_chat.tools import RetrievalTool

# output schema


class EnforcementOutputSchema(BaseModel):
    final_compliance_answer: str = Field(
        description="The absolute, verified legal roadmap formatted in clean Markdown, using tailored tone mechanics."
    )
    discarded_assumptions: List[str] = Field(
        default_factory=list,
        description="List of any Agent 1 assumptions that were legally incorrect and rejected during the statutory audit."
    )


@CrewBase
class EnforcementCrew():
    """Crew for handling Regulatory Compliance Verification, Auditing, and Final Output Generation"""

    # Points to your config folders where the YAMLs live
    # agents_config = 'config/agents.yaml'
    # tasks_config = 'config/tasks.yaml'
    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def compliance_enforcer(self) -> Agent:
        return Agent(
            config=self.agents_config['compliance_enforcer'],
            # tools avaialable
            tools=[RetrievalTool()],
            verbose=True,
            allow_delegation=False
        )

    @task
    def compliance_enforcement_task(self) -> Task:
        return Task(
            config=self.tasks_config['compliance_enforcement_task'],
            agent=self.compliance_enforcer(),
            # enforce schema
            output_json=EnforcementOutputSchema
        )

    @crew
    def crew(self) -> Crew:
        """Creates the isolated Enforcement Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=True
        )

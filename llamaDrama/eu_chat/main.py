# Main file, We have the conditional flow implemented here

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from crewai.flow.flow import Flow, start, listen, router

from llamaDrama.eu_chat.crew_files.crews.eu_specialist_crew.intake_triage_crew import TriageCrew
from llamaDrama.eu_chat.crew_files.crews.legal_expert_crew.compliance_crew import EnforcementCrew

# init what models llamaindex needs to use
from llamaDrama.eu_chat.init_llms import init_models

init_models()

# What Agent 1 needs send along to Agent 2


class AIActComplianceState(BaseModel):
    # Tracking inputs and iterations
    user_input: str = ""
    clarification_attempts: int = 0

    # Storage for Agent 1's structured payload
    is_sufficiently_narrow: bool = False
    clarification_question: Optional[str] = None
    role_extracted: Optional[str] = None
    jurisdiction_extracted: Optional[str] = None
    purpose_extracted: Optional[str] = None

    # Audience framing parameters
    technical_tier: Optional[str] = None
    organizational_role: Optional[str] = None
    primary_compliance_focus: Optional[str] = None

    # RAG parameters passed to Agent 2
    generated_subqueries: List[str] = Field(default_factory=list)
    agent_1_assumptions: List[str] = Field(default_factory=list)

    # Final Output Storage
    final_compliance_answer: str = ""
    discarded_assumptions: List[str] = Field(default_factory=list)


class EUAIActComplianceFlow(Flow[AIActComplianceState]):
    """
    Implements compliance flow.
    Keeps Agent 2 offline unless Agent 1 triggers a narrowness confirmation.
    """

    @start()
    def run_intake_and_triage(self):
        """Kicks off the Intake Specialist to parse the user query."""
        print(
            f"\n[Flow] Initializing Triage Step. Attempt Count: {self.state.clarification_attempts}")

        # Instantiate and run the isolated Agent 1 Crew
        triage_crew_instance = TriageCrew().crew()
        response = triage_crew_instance.kickoff(inputs={
            "user_input": self.state.user_input,
            "clarification_attempts": self.state.clarification_attempts
        })
        print("------AGENT 1 RAW RESPONSE:----------")
        print(response.raw)

        # CrewAI populates .raw or parsing can read raw string data
        try:
            if response.pydantic:
                data = response.pydantic
                self.state.is_sufficiently_narrow = data.is_sufficiently_narrow
                self.state.clarification_question = data.clarification_question
                self.state.role_extracted = data.role_extracted
                self.state.jurisdiction_extracted = data.jurisdiction_extracted
                self.state.purpose_extracted = data.purpose_extracted

                if data.audience_profile:
                    self.state.technical_tier = data.audience_profile.technical_tier
                    self.state.organizational_role = data.audience_profile.organizational_role
                    self.state.primary_compliance_focus = data.audience_profile.primary_compliance_focus

                self.state.generated_subqueries = data.generated_subqueries
                self.state.agent_1_assumptions = data.agent_1_assumptions

            else:
                # fallback:Crew AI populates.
                data_dict = response.json_dict if hasattr(
                    response, "json_dict") else json.loads(response.raw)

                self.state.is_sufficiently_narrow = data_dict.get(
                    "is_sufficiently_narrow", False)
                self.state.clarification_question = data_dict.get(
                    "clarification_question")
                self.state.role_extracted = data_dict.get("role_extracted")
                self.state.jurisdiction_extracted = data_dict.get(
                    "jurisdiction_extracted")
                self.state.purpose_extracted = data_dict.get(
                    "purpose_extracted")

                profile = data_dict.get("audience_profile", {}) or {}
                self.state.technical_tier = profile.get("technical_tier")
                self.state.organizational_role = profile.get(
                    "organizational_role")
                self.state.primary_compliance_focus = profile.get(
                    "primary_compliance_focus")

                self.state.generated_subqueries = data_dict.get(
                    "generated_subqueries", [])
                self.state.agent_1_assumptions = data_dict.get(
                    "agent_1_assumptions", [])

        except Exception as e:
            # This should catch System errors not Human ones
            print(f"[Error] Failed parsing Agent 1 JSON payload: {e}")
            # Safe defensive fallback: assume un-narrowed text to trigger a safe ask retry
            self.state.is_sufficiently_narrow = False
            # reset the state
            self.state.role_extracted = None
            self.state.jurisdiction_extracted = None
            self.state.purpose_extracted = None
            self.state.generated_subqueries = []
            self.state.agent_1_assumptions = []
            # This is a safe guard. SO in normal cases the code should NEVER come here
            self.state.clarification_question = (
                "I encountered a temporary formatting error while analyzing your profile. "
                "Could you restate your role, jurisdiction, and the purpose of your AI system clearly?"
            )

    @router(run_intake_and_triage)
    def evaluation_gate(self):
        """Evaluates the 3 Compliance Pillars and directs the processing route."""
        # 3 pillars
        pillars = [
            self.state.role_extracted,
            self.state.jurisdiction_extracted,
            self.state.purpose_extracted
        ]
        pillar_count = sum(1 for p in pillars if p and str(
            p).strip().lower() != "unknown")
        # Happy case
        if self.state.is_sufficiently_narrow or pillar_count >= 2:
            print(
                f"[Router] Sufficient compliance parameters met ({pillar_count}/3 pillars). Routing to Enforcer.")
            return "route_to_enforcer"
        # Uncooperative user case
        if self.state.clarification_attempts >= 2:
            print(
                f"[Router] Max loop attempts breached ({self.state.clarification_attempts} >= 2). Directing to Hard Exit.")
            return "route_to_hard_exit"

        # tell me more
        print(
            f"[Router] Insufficient parameters ({pillar_count}/3 pillars). Directing to Clarification Loop Path.")
        return "route_to_clarification_loop"

    @listen("route_to_enforcer")
    def run_compliance_enforcer(self):
        """Wakes up the optimized Compliance Enforcer to run targeted RAG auditing."""
        print("[Flow] Initializing Enforcer Step. Launching restricted RAG lookups.")

        # Instantiate and invoke the isolated Agent 2 Crew
        enforcer_crew_instance = EnforcementCrew().crew()
        response = enforcer_crew_instance.kickoff(inputs={
            # doesnt really need this. will have to adjust later
            "is_sufficiently_narrow": self.state.is_sufficiently_narrow,
            "generated_subqueries": self.state.generated_subqueries,
            "agent_1_assumptions": self.state.agent_1_assumptions,
            "technical_tier": self.state.technical_tier,
            "organizational_role": self.state.organizational_role,
            "primary_compliance_focus": self.state.primary_compliance_focus,
            "role_extracted": self.state.role_extracted,
            "jurisdiction_extracted": self.state.jurisdiction_extracted,
            "purpose_extracted": self.state.purpose_extracted
        })
        print("------RAW RESPONSE:----------")
        print(response.raw)

        try:
            if response.pydantic:
                self.state.final_compliance_answer = response.pydantic.final_compliance_answer
                self.state.discarded_assumptions = response.pydantic.discarded_assumptions
            else:
                # looking to see if that outpput was created by the Agent.
                if isinstance(getattr(response, "json_dict", None), dict):
                    data_dict = response.json_dict
                else:
                    try:
                        # attempt to parse the output
                        data_dict = json.loads(response.raw)
                    except (json.JSONDecodeError, TypeError):
                        # happens for Markdown text so collect it
                        data_dict = {
                            "final_compliance_answer": response.raw,
                            "discarded_assumptions": []
                        }

        except Exception as e:
            print(f"[Error] Failed parsing Agent 2 JSON payload: {e}")
            data_dict = {
                'final_compliance_answer': "Error generating finalized legal compliance map.",
                "discarded_assumptions": []
            }
            self.state.final_compliance_answer = "Error generating finalized legal compliance map."

        print("[Flow] Compliance Roadmap generated successfully. Ending pipeline.")
        self.state.final_compliance_answer = data_dict['final_compliance_answer']
        return self.state.final_compliance_answer

    @listen("route_to_clarification_loop")
    def process_clarification_turn(self):
        """Increments attempt counters and surfaces follow-up prompts to the UI loop."""
        self.state.clarification_attempts += 1
        print(
            f"[Flow] Surfacing question to UI. New Attempt Counter: {self.state.clarification_attempts}")

        # Assign the question text directly as the execution return value
        self.state.final_compliance_answer = self.state.clarification_question
        return self.state.final_compliance_answer

    @listen("route_to_hard_exit")
    def process_hard_exit(self):
        """Bypasses downstream nodes to push termination payloads directly to UI."""
        print("[Flow] Pipeline shut down via Hard Exit execution.")

        # Ensure the error reset string populates the terminal field
        self.state.final_compliance_answer = self.state.clarification_question
        return self.state.final_compliance_answer


if __name__ == "__main__":
    # Test Scenario 1: Complex, narrow query (Should bypass clarification directly)
    initial_prompt = "I am building a free app for a small charity to help homeless people find shelters in France. Does the EU AI Act apply to me, or are non-profits exempt since we aren't selling anything?"

    flow_execution = EUAIActComplianceFlow()
    flow_execution.state.user_input = initial_prompt
    flow_execution.state.clarification_attempts = 0

    output_result = flow_execution.kickoff()

    print("\n" + "="*40 + "\nFINAL WORKFLOW OUTPUT:\n" + "="*40)
    print(output_result)

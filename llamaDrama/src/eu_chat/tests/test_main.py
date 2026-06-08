import sys
from unittest.mock import MagicMock, patch
from main import EUAIActComplianceFlow, AIActComplianceState
from crew_files.crews.eu_specialist_crew.intake_triage_crew import TriageOutputSchema, AudienceProfile
from crew_files.crews.legal_expert_crew.compliance_crew import EnforcementOutputSchema


def execute_mocked_dry_run():
    print("=== Starting Dry Run Validation ===")

    # 1. Forge a dummy Agent 1 (Triage) output payload structure
    mock_triage_pydantic = TriageOutputSchema(
        is_sufficiently_narrow=True,
        clarification_question=None,
        role_extracted="Provider",
        jurisdiction_extracted="EU In-Scope",
        purpose_extracted="High-Risk",
        audience_profile=AudienceProfile(
            technical_tier="Technical",
            organizational_role="Lead Architect",
            primary_compliance_focus="Data Pipeline Governance"
        ),
        generated_subqueries=[
            "EU AI Act Article 9 data governance requirements"],
        agent_1_assumptions=[
            "System acts as a core training infrastructure dataset"]
    )

    # 2. Forge a dummy Agent 2 (Enforcer) output payload structure
    mock_enforcer_pydantic = EnforcementOutputSchema(
        final_compliance_answer="### Mocked Success Compliance Roadmap Text",
        discarded_assumptions=[]
    )

    # 3. Build CrewAI container response mock wrappers
    mock_triage_response = MagicMock()
    mock_triage_response.pydantic = mock_triage_pydantic

    mock_enforcer_response = MagicMock()
    mock_enforcer_response.pydantic = mock_enforcer_pydantic

    # 4. Bind context managers to intercept kickoff calls before they hit LLMs
    print("[Dry Run] Patching Crew execution layers...")
    with patch('crew_files.crews.eu_specialist_crew.intake_triage_crew.TriageCrew.crew') as mock_triage_crew, \
            patch('crew_files.crews.legal_expert_crew.compliance_crew.EnforcementCrew.crew') as mock_enforcer_crew:

        # Override the native kickoff function methods
        mock_triage_crew.return_value.kickoff = MagicMock(
            return_value=mock_triage_response)
        mock_enforcer_crew.return_value.kickoff = MagicMock(
            return_value=mock_enforcer_response)

        # 5. Bootstrap execution state exactly like main production runs
        flow_execution = EUAIActComplianceFlow()
        flow_execution.state.user_input = "Dry run verification string text."
        flow_execution.state.clarification_attempts = 0

        print("[Dry Run] Running workflow execution path...")
        try:
            output_result = flow_execution.kickoff()
            print("\n" + "="*40 + "\nDRY RUN WORKFLOW SUCCESSFUL:\n" + "="*40)
            print(f"Final Captured Return: {output_result}")
            print(
                f"State Pillar Count Verification: Role={flow_execution.state.role_extracted}, Jurisdiction={flow_execution.state.jurisdiction_extracted}")

        except Exception as flow_err:
            print(
                f"\n[CRITICAL RUNTIME ERROR] Flow architecture broke during execution: {flow_err}")
            sys.exit(1)


if __name__ == "__main__":
    execute_mocked_dry_run()

"""
Pydantic models for structured LLM output in Agent Design Process workflow.

These models define the schema for the 7-step agent design process:
1. PEAS Analysis (Performance, Environment, Actuators, Sensors)
2. Environment Properties Classification
3. Agent Function Specification
4. Agent Type Selection
5. ATDD Acceptance Criteria
6. Action Pair Design
7. Implementation Generation
"""

from typing import Literal

from pydantic import BaseModel, Field

# =============================================================================
# Step 1: PEAS Analysis (Russell & Norvig Framework)
# =============================================================================


class PerformanceMeasure(BaseModel):
    """Performance measure for the agent."""

    name: str = Field(description="Name of the metric")
    description: str = Field(description="What this measures")
    success_criteria: str = Field(description="How success is determined")
    measurable: bool = Field(
        default=True,
        description="Whether this metric can be objectively measured",
    )


class EnvironmentElement(BaseModel):
    """Element of the agent's environment."""

    name: str = Field(description="Environment component name")
    description: str = Field(description="Role in the environment")
    observable: bool = Field(description="Whether agent can observe this")
    modifiable: bool = Field(
        default=False,
        description="Whether agent can modify this element",
    )


class Actuator(BaseModel):
    """Agent actuator/action capability."""

    name: str = Field(description="Actuator name (action verb)")
    description: str = Field(description="What the agent can do")
    effect: str = Field(description="Effect on environment")
    category: Literal["external", "sensing", "internal"] = Field(
        default="external",
        description="Action category per agent function spec",
    )


class Sensor(BaseModel):
    """Agent sensor/percept capability."""

    name: str = Field(description="Sensor name")
    description: str = Field(description="What the agent perceives")
    data_type: str = Field(description="Type of data received")
    source: str = Field(
        default="",
        description="Which environment element this sensor observes",
    )


class PEASAnalysis(BaseModel):
    """
    Output of PEAS analysis step (Russell & Norvig framework).

    PEAS = Performance, Environment, Actuators, Sensors
    This is the foundational analysis for any intelligent agent design.
    """

    performance_measures: list[PerformanceMeasure] = Field(
        description="Metrics that define agent success"
    )
    environment_elements: list[EnvironmentElement] = Field(
        description="Components of the task environment"
    )
    actuators: list[Actuator] = Field(description="Actions the agent can perform")
    sensors: list[Sensor] = Field(description="Percepts the agent can receive")
    summary: str = Field(description="Brief summary of the agent's task environment")


# =============================================================================
# Step 2: Environment Properties Classification
# =============================================================================


class EnvironmentProperty(BaseModel):
    """Single environment property classification."""

    dimension: Literal[
        "observable",
        "deterministic",
        "static",
        "discrete",
        "agents",
        "known",
    ] = Field(description="The dimension being classified")
    classification: str = Field(
        description="The classification value (e.g., 'partially observable', 'stochastic')"
    )
    justification: str = Field(
        description="Why this classification was chosen based on PEAS"
    )


class EnvironmentPropertiesAnalysis(BaseModel):
    """
    Output of environment classification step.

    Classifies the task environment across 6 dimensions:
    1. Observable: fully vs partially observable
    2. Deterministic: deterministic vs stochastic
    3. Static: static vs dynamic
    4. Discrete: discrete vs continuous
    5. Agents: single-agent vs multi-agent
    6. Known: known vs unknown
    """

    properties: list[EnvironmentProperty] = Field(
        min_length=6,
        max_length=6,
        description="Classification for all 6 dimensions",
    )
    overall_complexity: Literal["simple", "moderate", "complex"] = Field(
        description="Overall environment complexity assessment"
    )
    key_challenges: list[str] = Field(
        description="Main challenges identified from environment properties"
    )


# =============================================================================
# Step 3: Agent Function Specification
# =============================================================================


class Percept(BaseModel):
    """A percept the agent can receive."""

    name: str = Field(description="Percept identifier")
    source: str = Field(description="Which sensor provides this")
    data_structure: str = Field(description="Expected data format/schema")
    example: str = Field(default="", description="Example percept value")


class Action(BaseModel):
    """An action the agent can take."""

    name: str = Field(description="Action identifier")
    category: Literal["external", "sensing", "internal"] = Field(
        description="Action category"
    )
    precondition: str = Field(
        description="When this action is applicable (ρ in ActionPair)"
    )
    effect: str = Field(description="What changes when action is taken")
    actuator: str = Field(description="Which actuator performs this action")


class PerceptActionPair(BaseModel):
    """A percept sequence leading to an action."""

    scenario_name: str = Field(description="Name describing this scenario")
    percepts: list[str] = Field(description="Sequence of percept names")
    action: str = Field(description="Resulting action name")
    rationale: str = Field(description="Why this mapping makes sense")


class AgentFunctionSpec(BaseModel):
    """
    Output of agent function definition step.

    Defines the agent function: f: P* → A
    Where P* is the percept sequence space and A is the action space.
    """

    percepts: list[Percept] = Field(description="All percepts the agent can receive")
    actions: list[Action] = Field(description="All actions the agent can take")
    percept_action_sequences: list[PerceptActionPair] = Field(
        description="Typical percept-action mappings"
    )
    state_representation: str = Field(
        description="How internal state is represented (if model-based)"
    )
    state_variables: list[str] = Field(
        default_factory=list,
        description="Key state variables maintained by the agent",
    )


# =============================================================================
# Step 4: Agent Type Selection
# =============================================================================


class AgentTypeAnalysis(BaseModel):
    """
    Output of agent type selection step.

    Selects from the Russell & Norvig agent taxonomy:
    - simple_reflex: Condition-action rules on current percept
    - model_based_reflex: Maintains internal state model
    - goal_based: Explicit goal representation, search
    - utility_based: Maximizes utility function
    - learning: Improves over time
    """

    selected_type: Literal[
        "simple_reflex",
        "model_based_reflex",
        "goal_based",
        "utility_based",
        "learning",
    ] = Field(description="The selected agent type")
    justification: str = Field(
        description="Why this type was selected based on environment and function"
    )
    alternatives_considered: list[str] = Field(
        description="Other agent types that were considered"
    )
    rejection_reasons: dict[str, str] = Field(
        description="Why each alternative was rejected"
    )
    required_capabilities: list[str] = Field(
        description="What the implementation must support for this type"
    )
    dual_state_rationale: str = Field(
        default="",
        description="How this maps to Dual-State architecture (S_workflow × S_env)",
    )


# =============================================================================
# Step 5: ATDD Acceptance Criteria
# =============================================================================


class AcceptanceScenario(BaseModel):
    """
    Single acceptance scenario in Given-When-Then format.

    Follows ATDD principles from the Agent Design Process v2.
    """

    scenario_id: str = Field(description="Unique identifier, e.g., 'AC-001'")
    name: str = Field(description="Descriptive scenario name")
    given: list[str] = Field(description="Preconditions - percepts and initial state")
    when: list[str] = Field(description="Trigger - the action or event being tested")
    then: list[str] = Field(description="Expected outcomes - observable behaviors")
    percept_refs: list[str] = Field(
        description="References to percepts from agent function"
    )
    action_refs: list[str] = Field(
        description="References to actions from agent function"
    )
    principle_compliance: dict[str, bool] = Field(
        default_factory=dict,
        description="Which of the 10 principles this scenario follows",
    )


class AcceptanceCriteria(BaseModel):
    """
    Output of ATDD step - acceptance criteria for the agent.

    These criteria bridge the agent function specification with
    implementation, following the 10 Principles for Acceptance Criteria.
    """

    scenarios: list[AcceptanceScenario] = Field(
        min_length=1,
        description="Given-When-Then acceptance scenarios",
    )
    coverage_summary: str = Field(
        description="Summary of which percept-action pairs are covered"
    )
    untested_behaviors: list[str] = Field(
        default_factory=list,
        description="Agent behaviors not covered by acceptance criteria",
    )


# =============================================================================
# Step 6: Action Pair Design
# =============================================================================


class ActionPairSpec(BaseModel):
    """
    Single Action Pair specification: A = ⟨ρ, a_gen, G⟩

    Per the Dual-State architecture:
    - ρ (rho): Precondition - when can this action be taken?
    - a_gen: Generator - what produces the output?
    - G: Guard - how do we verify success?
    """

    action_pair_id: str = Field(
        description="Unique identifier, e.g., 'ap_process_task'"
    )
    name: str = Field(description="Human-readable name")
    description: str = Field(description="What this action pair accomplishes")

    # ρ (rho) - Precondition
    precondition: str = Field(
        description="Boolean expression for when this action applies"
    )
    precondition_percepts: list[str] = Field(
        description="Percepts that must be present"
    )

    # a_gen - Generator
    generator_name: str = Field(description="Name of the generator class to create")
    generator_description: str = Field(description="What the generator produces")
    generator_inputs: list[str] = Field(description="Inputs the generator needs")
    generator_output_schema: str = Field(description="Description of output format")

    # G - Guard
    guard_name: str = Field(description="Name of the guard class to create")
    guard_description: str = Field(description="What the guard validates")
    guard_checks: list[str] = Field(description="Specific validation checks")

    # Traceability
    acceptance_criteria_refs: list[str] = Field(
        description="Which acceptance scenarios this implements"
    )
    action_refs: list[str] = Field(
        description="Which agent function actions this implements"
    )


class ActionPairsDesign(BaseModel):
    """
    Output of Action Pair design step.

    Defines the complete set of action pairs for the agent's workflow,
    following the Dual-State architecture pattern.
    """

    action_pairs: list[ActionPairSpec] = Field(
        min_length=1, description="All action pairs for the agent"
    )
    workflow_order: list[str] = Field(description="Execution order of action pair IDs")
    dependencies: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Dependencies between action pairs (DAG edges)",
    )
    state_transitions: str = Field(description="Description of workflow state machine")


# =============================================================================
# Step 7: Implementation Generation
# =============================================================================


class GeneratedFile(BaseModel):
    """A file to be generated."""

    path: str = Field(description="Relative path for the file")
    content: str = Field(description="File content")
    description: str = Field(description="Purpose of this file")
    file_type: Literal["python", "json", "markdown"] = Field(
        default="python",
        description="Type of file",
    )


class WorkflowStepConfig(BaseModel):
    """Configuration for a workflow step in the generated workflow.json."""

    step_id: str = Field(description="Step identifier")
    generator: str = Field(description="Generator class name")
    guard: str = Field(description="Guard identifier")
    requires: list[str] = Field(
        default_factory=list,
        description="Dependencies (other step IDs)",
    )
    description: str = Field(description="What this step does")


class ImplementationResult(BaseModel):
    """
    Output of implementation generation step.

    Produces the complete skeleton for a Dual-State Action Pair agent.
    """

    workflow_config: dict = Field(description="Content for workflow.json")
    workflow_steps: list[WorkflowStepConfig] = Field(
        description="Workflow step configurations"
    )
    files: list[GeneratedFile] = Field(description="Generated skeleton files")
    setup_instructions: str = Field(description="How to run the generated agent")
    design_summary: str = Field(description="Summary of the agent design")

from src.graph.state import State


def route_on_confidence(state: State) -> str:
    """
    Route the workflow based on a confidence/evidence decision that has already been computed.

    Returns:
        "generate" when the evidence is sufficient, otherwise "insufficient".
    """
    if state.evidence_sufficient:
        return "generate"

    return "insufficient"

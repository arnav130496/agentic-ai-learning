from src.graph.state import State
from langgraph.graph import StateGraph, START, END
from src.graph.nodes import score_confidence, generate_constrained, insufficient_evidence, attach_citations
from src.graph.routing import route_on_confidence

builder = StateGraph(State)
builder.add_node(score_confidence)
builder.add_node(insufficient_evidence)
builder.add_node(generate_constrained)
builder.add_node(attach_citations)

builder.add_edge(START, "score_confidence")
builder.add_conditional_edges("score_confidence", route_on_confidence, {"generate": "generate_constrained", "insufficient": "insufficient_evidence"})
builder.add_edge("generate_constrained", "attach_citations")
builder.add_edge("insufficient_evidence", END)
builder.add_edge("attach_citations", END)

graph = builder.compile()
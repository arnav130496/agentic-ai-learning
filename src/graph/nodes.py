from src.graph.state import State, ConstrainedAnswer
from src.graph.model import get_llm
from langchain_core.prompts import ChatPromptTemplate

def score_confidence(state: State) -> dict:
    """
    This function takes a State object as input and calculates the confidence score based on the retrieved chunks.
    """
    retrieved_chunks = state.retrieved_chunks

    if not retrieved_chunks:
        return {"confidence_score": 0.0, "evidence_sufficient": False}
    
    top_scores = sorted(
        (chunk["rerank_score"] for chunk in retrieved_chunks),
        reverse=True,
    )[:3]

    confidence_score = sum(top_scores) / len(top_scores)
    threshold = 0.5
    evidence_sufficient = confidence_score >= threshold

    return {"confidence_score": confidence_score, "evidence_sufficient": evidence_sufficient}

def generate_constrained(state: State) -> dict:
    """
    This function takes a State object as input and generates a constrained answer based on the retrieved chunks.
    """
    llm = get_llm()
    structred_model = llm.with_structured_output(ConstrainedAnswer)
    context = "\n\n".join(chunk["text"] for chunk in state.retrieved_chunks)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer only using the provided context. If the context doesn't contain the answer, say so.\n\nContext:\n{context}"),
        ("user", "{query}")
    ])
    
    chain = prompt | structred_model
    response = chain.invoke({"context": context, "query": state.query})

    return {"answer": response.answer}


def insufficient_evidence(state: State) -> dict:
    """
    This function takes a State object as input and checks if the retrieved evidence is sufficient to answer the query.
    """
    if state.evidence_sufficient:
        return {"answer": "Evidence is sufficient to answer the query."}
    else:
        return {"answer": "Evidence is insufficient to answer the query.", "citations": []}

def attach_citations(state: State) -> dict:
    """
    This function takes a State object as input and attaches citations to the answer based on the retrieved chunks.
    """
    retrieved_chunks = state.retrieved_chunks
    sources = list()
    for chunk in retrieved_chunks:
        if "source" in chunk:
            sources.append(chunk["chunk_id"] + " : " + chunk["source"])
    return {"citations": sources}
    

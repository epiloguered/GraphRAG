import ast
from typing import Any, Dict, List


def prepare_subquery_visualization(sub_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes = [{"id": "original", "name": "Original Question", "category": "question", "symbolSize": 40}]
    links = []

    for index, sub_question in enumerate(sub_questions):
        sub_id = f"sub_{index}"
        label = sub_question.get("sub-question", "") or sub_question.get("question", "")
        nodes.append({
            "id": sub_id,
            "name": (label[:32] + "...") if len(label) > 32 else label,
            "category": "sub_question",
            "symbolSize": 28,
        })
        links.append({"source": "original", "target": sub_id, "name": "decomposed_to"})

    return {
        "nodes": nodes,
        "links": links,
        "categories": [
            {"name": "question", "itemStyle": {"color": "#ff6b6b"}},
            {"name": "sub_question", "itemStyle": {"color": "#4ecdc4"}},
        ],
    }


def prepare_reasoning_flow_visualization(reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = []
    for index, step in enumerate(reasoning_steps):
        steps.append({
            "step": index + 1,
            "type": step.get("type", "unknown"),
            "question": step.get("question", "")[:80],
            "triples_count": step.get("triples_count", 0),
            "chunks_count": step.get("chunks_count", 0),
            "processing_time": step.get("processing_time", 0),
        })

    return {
        "steps": steps,
        "timeline": [item["processing_time"] for item in steps],
    }


def subgraph_from_triples(triples: List[str], source_strategy: str = "unknown") -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    node_index: Dict[str, Dict[str, Any]] = {}
    categories = {"entity"}

    for triple in triples:
        try:
            parts = None
            if triple.startswith("[") and triple.endswith("]"):
                parsed = ast.literal_eval(triple)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                    parts = [str(parsed[0]), str(parsed[1]), str(parsed[2])]
            else:
                body = triple
                if " [score:" in body:
                    body = body.split(" [score:", 1)[0]
                if body.startswith("(") and body.endswith(")"):
                    body = body[1:-1]
                parsed = [piece.strip() for piece in body.split(", ", 2)]
                if len(parsed) == 3:
                    parts = parsed
            if not parts:
                continue
            source, relation, target = parts
            for entity in (source, target):
                if entity not in node_index:
                    node = {
                        "id": entity,
                        "name": entity[:32],
                        "category": "entity",
                        "symbolSize": 20,
                        "source_strategy": source_strategy,
                        "first_seen_step": 1,
                    }
                    node_index[entity] = node
                    nodes.append(node)
            links.append({
                "source": source,
                "target": target,
                "name": relation,
                "relation": relation,
                "source_strategy": source_strategy,
                "first_seen_step": 1,
            })
        except Exception:
            continue

    return {
        "nodes": nodes,
        "links": links,
        "categories": [{"name": name, "itemStyle": {"color": "#95de64"}} for name in sorted(categories)],
    }


def prepare_strategy_visualization(result: Dict[str, Any]) -> Dict[str, Any]:
    reasoning_subgraph = result.get("reasoning_subgraph") or subgraph_from_triples(
        result.get("retrieved_triples", []),
        source_strategy=result.get("strategy_name", "unknown"),
    )

    return {
        "subqueries": prepare_subquery_visualization(result.get("sub_questions", [])),
        "knowledge_graph": reasoning_subgraph,
        "reasoning_flow": prepare_reasoning_flow_visualization(result.get("reasoning_steps", [])),
        "retrieval_details": {
            "total_triples": len(result.get("retrieved_triples", [])),
            "total_chunks": len(result.get("retrieved_chunks", [])),
            "sub_questions_count": len(result.get("sub_questions", [])),
            "triples_by_subquery": [
                step.get("triples_count", 0)
                for step in result.get("reasoning_steps", [])
                if step.get("type") == "sub_question"
            ],
        },
    }

"""
kg.py - small knowledge graph for the 7 HAM10000 lesion classes.

Nodes:  condition / feature / risk / care
Edges:  has_feature, risk_factor, confused_with, seek_care_if

IMPORTANT: the medical content below is a starting draft written from
general dermatology knowledge. Before relying on it, verify every entry
against a reputable source (AAD, NCI, DermNet, NHS) and put the source
in SOURCES / the edge "source" attribute. Keep it short and conservative.
"""

import networkx as nx


# ------------------------------------------------------------------
# CURATED CONTENT  (edit this section, not the code below)
# ------------------------------------------------------------------

# code -> (display name, one-line description)
CONDITIONS = {
    "akiec": (
        "Actinic Keratosis",
        "A rough, scaly sun-damaged patch of skin that is considered "
        "precancerous. In HAM10000 this class also includes Bowen's disease.",
    ),
    "bcc": (
        "Basal Cell Carcinoma",
        "The most common type of skin cancer. It usually grows slowly and "
        "rarely spreads, but it needs medical treatment.",
    ),
    "bkl": (
        "Benign Keratosis",
        "A group of non-cancerous growths, such as seborrheic keratoses "
        "and sun-related lentigines.",
    ),
    "df": (
        "Dermatofibroma",
        "A common, harmless firm skin nodule.",
    ),
    "mel": (
        "Melanoma",
        "A serious type of skin cancer that starts in pigment-producing "
        "cells. Early detection matters.",
    ),
    "nv": (
        "Nevus",
        "A common mole made of pigment-producing cells. Most are harmless.",
    ),
    "vasc": (
        "Vascular Lesion",
        "A lesion made of small blood vessels, such as a cherry angioma. "
        "Most are harmless.",
    ),
}

# code -> list of typical appearance features
FEATURES = {
    "akiec": ["rough, scaly patch", "pink, red or brown colour",
              "found on sun-exposed skin"],
    "bcc": ["pearly or translucent bump", "visible small blood vessels",
            "sore that does not heal or keeps returning"],
    "bkl": ["waxy or 'stuck-on' look", "tan to dark brown colour",
            "sharply defined edge"],
    "df": ["small firm nodule", "pink to brown colour",
           "often on the legs"],
    "mel": ["asymmetric shape", "irregular border",
            "more than one colour", "diameter larger than about 6 mm",
            "changing over time"],
    "nv": ["symmetric shape", "even colour", "smooth, well-defined border"],
    "vasc": ["red, purple or blue colour", "small, well-defined spot"],
}

# code -> list of risk factors
RISKS = {
    "akiec": ["long-term sun (UV) exposure", "fair skin", "older age"],
    "bcc": ["long-term sun (UV) exposure", "fair skin", "history of sunburns"],
    "bkl": ["older age"],
    "df": [],
    "mel": ["long-term sun (UV) exposure", "history of sunburns", "fair skin",
            "many moles", "family history of melanoma"],
    "nv": ["genetic tendency to have many moles",
           "long-term sun (UV) exposure"],
    "vasc": ["older age"],
}

# (code_a, code_b): pairs that are commonly mistaken for each other.
# Stored in both directions.
CONFUSED_WITH = [
    ("mel", "nv"),
    ("mel", "bkl"),
    ("akiec", "bkl"),
    ("akiec", "bcc"),
    ("df", "nv"),
]

# code -> when to seek medical attention
SEEK_CARE = {
    "akiec": "See a doctor; untreated patches can sometimes progress.",
    "bcc": "See a doctor for examination and treatment.",
    "bkl": "See a doctor if it changes quickly, bleeds, or you are unsure.",
    "df": "See a doctor if it changes, bleeds, or you are unsure.",
    "mel": "See a doctor promptly for professional evaluation.",
    "nv": "See a doctor if it changes in size, shape or colour, or bleeds.",
    "vasc": "See a doctor if it bleeds repeatedly or grows quickly.",
}

# Always added to the context so explanations state the tool's scope.
LIMITATION = (
    "Limitation: this AI tool only covers 7 lesion categories. The lesion "
    "could be a condition it does not cover, such as eczema, psoriasis, "
    "a fungal infection or another type of skin cancer."
)

# Fill these in as you verify each fact, e.g. {"mel": "AAD - Melanoma"}
SOURCES = {code: "TODO: verify" for code in CONDITIONS}


# ------------------------------------------------------------------
# BUILD GRAPH
# ------------------------------------------------------------------

def build_graph():
    g = nx.DiGraph()

    for code, (name, desc) in CONDITIONS.items():
        cnode = f"condition:{code}"
        g.add_node(cnode, kind="condition", code=code, name=name, desc=desc)

        for feat in FEATURES.get(code, []):
            g.add_node(f"feature:{feat}", kind="feature", name=feat)
            g.add_edge(cnode, f"feature:{feat}",
                       relation="has_feature", source=SOURCES[code])

        # risk nodes are shared, e.g. UV exposure links several conditions
        for risk in RISKS.get(code, []):
            g.add_node(f"risk:{risk}", kind="risk", name=risk)
            g.add_edge(cnode, f"risk:{risk}",
                       relation="risk_factor", source=SOURCES[code])

        care = SEEK_CARE.get(code)
        if care:
            g.add_node(f"care:{code}", kind="care", name=care)
            g.add_edge(cnode, f"care:{code}",
                       relation="seek_care_if", source=SOURCES[code])

    for a, b in CONFUSED_WITH:
        g.add_edge(f"condition:{a}", f"condition:{b}",
                   relation="confused_with", source="TODO: verify")
        g.add_edge(f"condition:{b}", f"condition:{a}",
                   relation="confused_with", source="TODO: verify")

    return g


GRAPH = build_graph()


# ------------------------------------------------------------------
# LOOKUP
# ------------------------------------------------------------------

def _neighbors(code, relation):
    """Names of nodes reached from a condition via one relation type."""
    node = f"condition:{code}"
    return [
        GRAPH.nodes[n]["name"]
        for _, n, data in GRAPH.out_edges(node, data=True)
        if data["relation"] == relation
    ]


def normalize_code(value):
    """Accept 'mel', 'Melanoma', 'melanoma' ... and return the class code."""
    if not value:
        return None
    v = str(value).strip().lower()
    if v in CONDITIONS:
        return v
    for code, (name, _) in CONDITIONS.items():
        if name.lower() == v:
            return code
    return None


def get_condition_context(condition, uncertain=False):
    """
    Return a short text block describing the condition and its
    graph neighbours, ready to paste into an LLM prompt.
    Returns "" if the condition is unknown.
    """
    code = normalize_code(condition)
    if code is None:
        return ""

    name, desc = CONDITIONS[code]
    lines = []

    if uncertain:
        lines.append(
            f"The AI result was UNCERTAIN. Its closest guess was {name}, "
            "but this guess is not reliable."
        )

    lines.append(f"Condition: {name}. {desc}")

    features = _neighbors(code, "has_feature")
    if features:
        lines.append("Typical appearance: " + "; ".join(features) + ".")

    risks = _neighbors(code, "risk_factor")
    if risks:
        lines.append("Risk factors: " + "; ".join(risks) + ".")

    lookalikes = _neighbors(code, "confused_with")
    if lookalikes:
        lines.append("Can be mistaken for: " + ", ".join(lookalikes) + ".")

    care = _neighbors(code, "seek_care_if")
    if care:
        lines.append("Medical advice: " + care[0])

    lines.append(LIMITATION)

    return "\n".join(lines)


def get_related_conditions(condition):
    """Codes of conditions that share a risk factor or look-alike link."""
    code = normalize_code(condition)
    if code is None:
        return []
    start = f"condition:{code}"
    related = set()

    for _, n, d in GRAPH.out_edges(start, data=True):
        if d["relation"] == "confused_with":
            related.add(GRAPH.nodes[n]["code"])
        elif d["relation"] == "risk_factor":
            for other, _, d2 in GRAPH.in_edges(n, data=True):
                if d2["relation"] == "risk_factor" and other != start:
                    related.add(GRAPH.nodes[other]["code"])
    return sorted(related)


def get_subgraph(condition):
    """
    1-hop neighbourhood of a condition as plain JSON-friendly data,
    for drawing in the frontend. The long 'seek care' text is left out.
    Risk nodes also list the other conditions that share them.
    """
    code = normalize_code(condition)
    if code is None:
        return None

    center = f"condition:{code}"
    nodes = {
        center: {
            "id": center, "label": CONDITIONS[code][0],
            "kind": "condition", "code": code, "shared_with": [],
        }
    }
    edges = []

    for _, n, data in GRAPH.out_edges(center, data=True):
        relation = data["relation"]
        if relation == "seek_care_if":
            continue

        nd = GRAPH.nodes[n]
        shared = []
        if nd["kind"] == "risk":
            shared = sorted(
                GRAPH.nodes[o]["name"]
                for o, _, d2 in GRAPH.in_edges(n, data=True)
                if d2["relation"] == "risk_factor" and o != center
            )

        nodes[n] = {
            "id": n, "label": nd["name"], "kind": nd["kind"],
            "code": nd.get("code"), "shared_with": shared,
        }
        edges.append({"source": center, "target": n, "relation": relation})

    return {"code": code, "center": center,
            "nodes": list(nodes.values()), "edges": edges}


if __name__ == "__main__":
    print(f"Graph: {GRAPH.number_of_nodes()} nodes, "
          f"{GRAPH.number_of_edges()} edges\n")
    print(get_condition_context("mel"))
    print("\n--- uncertain, top guess 'Nevus' ---")
    print(get_condition_context("Nevus", uncertain=True))
    print("\nRelated to mel:", get_related_conditions("mel"))
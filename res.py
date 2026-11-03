import sys
import ollama
import numpy as np

# =====================================================
# CONFIGURATION
# =====================================================

EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2"

MAX_DEPTH = 8
MIN_REGION_WORDS = 500
BOUNDARY_WORDS = 400
CONFIDENCE_THRESHOLD = 0.7
EMBED_CHUNK_WORDS = 256

# =====================================================
# COSINE SIMILARITY
# =====================================================

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# =====================================================
# EMBEDDING HELPERS
# =====================================================

def embed_small(text):
    response = ollama.embed(
        model=EMBED_MODEL,
        input=text
    )
    return np.array(response["embeddings"][0])

# =====================================================
# BOUNDARY VIEW
# =====================================================

def boundary_view(text, boundary_words=BOUNDARY_WORDS):
    """
    Returns beginning + ending portions of a region.
    This exposes the outer boundaries of the current context.
    """
    words = text.split()
    if len(words) <= boundary_words * 2:
        return text

    start = words[:boundary_words]
    end = words[-boundary_words:]
    return " ".join(start + end)

# =====================================================
# RECURSIVE BOUNDARY SPLIT (RBER Step 1)
# =====================================================

def split_three(text):
    """
    Splits text into outer boundaries (Left, Right) and inner core (Middle).
    Left  = Top Bun / Outer Start Boundary
    Right = Bottom Bun / Outer End Boundary
    Middle = Patty / Inner Region to expose
    """
    words = text.split()
    n = len(words)
    if n < 3:
        return text, "", ""

    third = n // 3
    left = " ".join(words[:third])
    middle = " ".join(words[third:2 * third])
    right = " ".join(words[2 * third:])

    return left, middle, right

# =====================================================
# RELEVANCE SCORING (RBER Step 2)
# =====================================================

def compute_relevance(q_emb, text):
    """
    Computes maximum semantic alignment between query and boundary chunks.
    """
    words = text.split()
    if not words:
        return 0.0
    max_sim = 0.0
    for i in range(0, len(words), EMBED_CHUNK_WORDS):
        chunk = " ".join(words[i:i + EMBED_CHUNK_WORDS])
        c_emb = embed_small(chunk)
        sim = cosine_similarity(q_emb, c_emb)
        if sim > max_sim:
            max_sim = sim
    return float(max_sim)

# =====================================================
# BOUNDARY EVALUATOR & CONFIDENCE (RBER Steps 4 & 5)
# =====================================================

def evaluate_boundary(text, question):
    """
    Boundary Evaluator:
    Evaluates whether the boundary segment contains the answer or secret key.
    Returns: (confidence, answer_or_summary)
    """
    view = boundary_view(text)
    system_msg = (
        "You are a precise document retrieval assistant. "
        "Your task is to answer the question using the provided context. "
        "Pay special attention to any explicit notes, combinations, secrets, keys, or numbers. "
        "If the context contains the answer, output the exact answer directly. "
        "If the context contains NO information to answer the question, output exactly: NOT_FOUND"
    )
    prompt = f"Context:\n{view}\n\nQuestion: {question}"

    response = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.0}
    )
    ans = response["message"]["content"].strip()
    cleaned = ans.upper()

    if cleaned == "NOT_FOUND" or cleaned.startswith("NOT_FOUND"):
        return 0.0, "NOT_FOUND"
    else:
        return 1.0, ans

# =====================================================
# FINAL ANSWER SYNTHESIS (RBER Step 9)
# =====================================================

def answer_question(question, current_region, evidence):
    """
    Synthesizes the final answer using Answer Context:
    Answer Context = Top Evidence + Current Region + Query
    """
    evidence_lines = []
    for item in evidence[-5:]:
        evidence_lines.append(
            f"[{item['segment_id']}]: {item.get('summary', '')}"
        )
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "None"

    system_msg = (
        "You are a precise document QA assistant. "
        "Answer the question based strictly on the supplied context and evidence. "
        "If a secret key, note, or combination is present, state it directly. "
        "If not found, respond exactly: NOT_FOUND"
    )
    prompt = f"Context:\n{current_region}\n\nAccumulated Evidence:\n{evidence_text}\n\nQuestion: {question}"

    response = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.0}
    )
    return response["message"]["content"].strip()

# =====================================================
# RECURSIVE BOUNDARY EXPOSURE RETRIEVAL (RBER Algorithm)
# =====================================================

def rber(document, question, depth=0, evidence=None):
    """
    Algorithm 1: Recursive Boundary Exposure Retrieval (RBER)
    Paper: "Recursive Boundary Exposure Retrieval (RBER): Exploiting Positional Bias for Long-Context Retrieval"
    
    Procedure:
    1. Extract boundary segments (Left and Right).
    2. Evaluate relevance to Q.
    3. Store evidence into repository.
    4. Compute confidence score C = P(answer | segment).
    5. If confidence > tau:
         Return evidence / answer immediately (Adaptive early stopping).
    6. Else:
         Remove low-value boundaries (Peel outer layers, exposing middle content).
    7. Recurse on remaining region.
    8. Merge evidence.
    9. Generate answer.
    """
    if evidence is None:
        evidence = []

    words = document.split()
    print(f"\n--- Depth {depth} | Current Region Size: {len(words)} words ---")

    # -----------------------------------
    # STOPPING CONDITION: Region Size or Max Depth
    # -----------------------------------
    if len(words) <= MIN_REGION_WORDS or depth >= MAX_DEPTH:
        print("\nReached isolated central core (Patty). Performing final evaluation...")
        return answer_question(question, document, evidence)

    # -----------------------------------
    # STEP 1: EXTRACT BOUNDARY SEGMENTS
    # -----------------------------------
    left, middle, right = split_three(document)

    # -----------------------------------
    # STEP 2: RELEVANCE SCORING
    # -----------------------------------
    q_emb = embed_small(question)
    rel_left = compute_relevance(q_emb, left)
    rel_right = compute_relevance(q_emb, right)

    print(f"Checking boundaries: LEFT relevance = {rel_left:.4f} | RIGHT relevance = {rel_right:.4f}")

    # -----------------------------------
    # STEP 3 & 4: EVALUATE BOUNDARIES & CONFIDENCE
    # -----------------------------------
    # Evaluate Left Boundary
    conf_left, ans_left = evaluate_boundary(left, question)

    # Store Left Evidence
    evidence.append({
        "depth": depth,
        "segment_id": f"D{depth}_LEFT",
        "relevance_score": rel_left,
        "confidence": conf_left,
        "summary": ans_left if conf_left > 0 else f"Inspected left boundary ({len(left.split())} words)"
    })

    # STEP 5: If confidence > tau, return evidence/answer
    if conf_left >= CONFIDENCE_THRESHOLD:
        print(f"\n>>> TARGET ISOLATED AT LEFT BOUNDARY (Depth {depth}) <<<")
        print(f"Confidence: {conf_left:.2f}")
        return ans_left

    # Evaluate Right Boundary
    conf_right, ans_right = evaluate_boundary(right, question)

    # Store Right Evidence
    evidence.append({
        "depth": depth,
        "segment_id": f"D{depth}_RIGHT",
        "relevance_score": rel_right,
        "confidence": conf_right,
        "summary": ans_right if conf_right > 0 else f"Inspected right boundary ({len(right.split())} words)"
    })

    # STEP 5: If confidence > tau, return evidence/answer
    if conf_right >= CONFIDENCE_THRESHOLD:
        print(f"\n>>> TARGET ISOLATED AT RIGHT BOUNDARY (Depth {depth}) <<<")
        print(f"Confidence: {conf_right:.2f}")
        return ans_right

    # -----------------------------------
    # STEP 6 & 7: PEEL BOUNDARIES & RECURSE ON MIDDLE
    # -----------------------------------
    print(f"Target not at outer boundaries at Depth {depth}. Peeling outer layers to expose middle ({len(middle.split())} words)...")
    return rber(middle, question, depth + 1, evidence)

# =====================================================
# MAIN ENTRYPOINT
# =====================================================

if __name__ == "__main__":
    with open("long.txt", "r", encoding="utf-8") as f:
        document = f.read()

    default_q = "Based on the above text what is the secret to the safe?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        try:
            user_input = input("\nQuestion: ").strip()
            question = user_input if user_input else default_q
        except (EOFError, KeyboardInterrupt):
            question = default_q

    print(f"\nQuestion: {question}")

    answer = rber(document, question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)
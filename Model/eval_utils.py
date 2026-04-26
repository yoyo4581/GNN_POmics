import torch
from Model.wrappers import build_explainer

def evaluate(model, validation_loader):
    # ── Stage 2: Post-hoc GraphMask explanation ───────────────────
    #
    # GraphMaskExplainer freezes the backbone and trains gate networks.
    # Each call to explainer() trains and then returns the edge_mask.
    #
    print("\n=== Stage 2: GraphMask explanation ===")
    model.eval()

    # build_explainer pre-computes x_normed and bakes batch in as buffers,
    # so the explainer call only needs x and edge_index (x is ignored internally
    # in favour of the pre-normalized buffer, but required by the Explainer API).
    explainer = build_explainer(model)

    for data in validation_loader:

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
        )

        print(f"\nEdge mask shape  : {explanation.edge_mask.shape}")
        print(f"Edges retained   : {(explanation.edge_mask > 0.5).sum().item()} / {data.edge_index.size(1)}")
        print(f"Top-5 edge scores: {explanation.edge_mask.topk(5).values.tolist()}")

        # ── Inference ────────────────────────────────────────────────
        print("\n=== Inference ===")
        results = model.predict(data.x, data.edge_index, data.batch, confidence_threshold=0.5)
        for r in results:
            flag = "" if r["reliable"] else "  ⚠ LOW CONFIDENCE — possible OOD"
            print(f"  → {r['tissue']}  conf={r['confidence']:.3f}{flag}")


        print("\nDone.")
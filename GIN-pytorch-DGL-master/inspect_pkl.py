import argparse
import pickle
from pathlib import Path
from pprint import pprint


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect and print content of a .pkl file")
    parser.add_argument(
        "--pkl",
        type=str,
        default="data/NCI_balanced/nci_malbehavd.pkl",
        help="Path to .pkl file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many items to print for list/tuple containers",
    )
    parser.add_argument(
        "--show-full",
        action="store_true",
        help="Print full object without truncating by --limit",
    )
    return parser.parse_args()


def print_graph_preview(graph):
    keys = sorted(graph.keys())
    print(f"keys: {keys}")
    for k in ("id", "graph_label", "number_node", "number_edge"):
        if k in graph:
            print(f"{k}: {graph[k]}")

    if "node_label" in graph:
        node_labels = graph["node_label"]
        print(f"node_label[:10]: {node_labels}")
    if "edge" in graph:
        edges = graph["edge"]
        print(f"edge[:10]: {edges[:10]}")
    if "edge_weight" in graph:
        edge_w = graph["edge_weight"]
        print(f"edge_weight[:10]: {edge_w[:10]}")


def inspect_object(obj, limit, show_full):
    print(f"type: {type(obj)}")
    if hasattr(obj, "__len__"):
        try:
            print(f"len: {len(obj)}")
        except Exception:
            pass

    if show_full:
        pprint(obj, width=120, compact=False)
        return

    if isinstance(obj, dict):
        print("dict preview:")
        for i, (k, v) in enumerate(obj.items(), start=1):
            print(f"- {k}: {type(v)}")
            if i >= limit:
                break
        return

    if isinstance(obj, (list, tuple)):
        n = min(limit, len(obj))
        print(f"showing first {n} item(s):")
        for i in range(n):
            item = obj[i]
            print(f"\n[{i}] type: {type(item)}")
            if isinstance(item, dict) and {"node_label", "edge", "edge_weight"}.issubset(item.keys()):
                print_graph_preview(item)
            else:
                pprint(item, width=120, compact=False)
        return

    pprint(obj, width=120, compact=False)


def main():
    args = parse_args()
    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        raise FileNotFoundError(f".pkl file not found: {pkl_path}")

    with open(pkl_path, "rb") as fp:
        obj = pickle.load(fp)

    inspect_object(obj, limit=args.limit, show_full=args.show_full)


if __name__ == "__main__":
    main()

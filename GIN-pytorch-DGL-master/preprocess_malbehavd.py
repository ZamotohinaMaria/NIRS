import argparse
import csv
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MalBehavD-V1 CSV into NCI-like pickle format used by loadNCI.py"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="../MalbehavD-V1-main/MalBehavD-V1-dataset.csv",
        help="Path to MalBehavD-V1 CSV file",
    )
    parser.add_argument(
        "--output-pkl",
        type=str,
        default="data/NCI_balanced/nci_malbehavd.pkl",
        help="Output pickle path",
    )
    parser.add_argument(
        "--label-format",
        type=str,
        default="pm1",
        choices=["pm1", "01"],
        help="Encode graph_label as -1/+1 (pm1) or 0/1 (01)",
    )
    parser.add_argument(
        "--edge-mode",
        type=str,
        default="undirected",
        choices=["undirected", "directed"],
        help="Build transition edges as undirected or directed before saving",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for quick debugging (0 means all rows)",
    )
    parser.add_argument(
        "--id-policy",
        type=str,
        default="make-unique",
        choices=["make-unique", "keep", "error"],
        help="How to handle duplicate sample ids (sha256)",
    )
    return parser.parse_args()


def normalize_label(raw_label, label_format):
    label_int = int(raw_label)
    if label_int not in (0, 1):
        raise ValueError(f"Expected binary label 0/1, got {raw_label}")
    if label_format == "pm1":
        return 1 if label_int == 1 else -1
    return label_int


def extract_api_calls(row):
    return [token.strip() for token in row[2:] if token and token.strip()]


def build_graph_from_calls(api_calls, edge_mode):
    node_to_id = {}
    node_labels = []
    for api in api_calls:
        if api not in node_to_id:
            # Store 1-based ids because loadNCI.py later subtracts 1.
            node_to_id[api] = len(node_labels) + 1
            node_labels.append(api)

    edge_weights = defaultdict(int)
    for src_api, dst_api in zip(api_calls, api_calls[1:]):
        src_id = node_to_id[src_api]
        dst_id = node_to_id[dst_api]
        if edge_mode == "undirected" and src_id > dst_id:
            src_id, dst_id = dst_id, src_id
        edge_weights[(src_id, dst_id)] += 1

    edge_list = list(edge_weights.keys())
    weight_list = [edge_weights[edge] for edge in edge_list]
    return node_labels, edge_list, weight_list


def resolve_sample_id(sample_id, row_index, id_counter, id_policy):
    base_id = sample_id if sample_id else f"sample_{row_index}"
    id_counter[base_id] += 1
    seen_count = id_counter[base_id]

    if seen_count == 1:
        return base_id
    if id_policy == "keep":
        return base_id
    if id_policy == "error":
        raise ValueError(f"Duplicate sample id '{base_id}' at CSV row {row_index}")
    return f"{base_id}__dup{seen_count}"


def validate_graph(graph, row_index):
    required = {"id", "graph_label", "number_node", "node_label", "number_edge", "edge", "edge_weight"}
    missing = required - set(graph.keys())
    if missing:
        raise ValueError(f"Missing keys {sorted(missing)} at row {row_index}")

    if graph["number_node"] != len(graph["node_label"]):
        raise ValueError(f"number_node mismatch at row {row_index}")
    if graph["number_edge"] != len(graph["edge"]):
        raise ValueError(f"number_edge mismatch at row {row_index}")
    if graph["number_edge"] != len(graph["edge_weight"]):
        raise ValueError(f"edge_weight length mismatch at row {row_index}")
    if graph["graph_label"] not in (-1, 0, 1):
        raise ValueError(f"Unexpected graph_label={graph['graph_label']} at row {row_index}")

    n = graph["number_node"]
    for edge_index, edge in enumerate(graph["edge"]):
        if not (isinstance(edge, (tuple, list)) and len(edge) == 2):
            raise ValueError(f"Bad edge format at row {row_index}, edge #{edge_index}: {edge}")
        src, dst = edge
        if not (isinstance(src, int) and isinstance(dst, int)):
            raise ValueError(f"Non-int edge index at row {row_index}, edge #{edge_index}: {edge}")
        if not (1 <= src <= n and 1 <= dst <= n):
            raise ValueError(
                f"Edge index out of range at row {row_index}, edge #{edge_index}: {edge}, node_count={n}"
            )

    for weight_index, weight in enumerate(graph["edge_weight"]):
        if not (isinstance(weight, int) and weight > 0):
            raise ValueError(
                f"Edge weight must be positive int at row {row_index}, edge_weight #{weight_index}: {weight}"
            )


def convert_csv(input_csv, label_format, edge_mode, limit, id_policy):
    graphs = []
    label_counter = Counter()
    id_counter = Counter()
    skipped = 0

    with open(input_csv, "r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        _ = next(reader, None)  # header

        for row_index, row in enumerate(reader, start=1):
            if limit > 0 and len(graphs) >= limit:
                break
            if len(row) < 3:
                skipped += 1
                continue

            raw_sample_id = row[0].strip()
            try:
                graph_label = normalize_label(row[1], label_format)
            except Exception as exc:
                raise ValueError(f"Invalid label at CSV row {row_index}: {row[1]!r}") from exc
            api_calls = extract_api_calls(row)
            if len(api_calls) < 2:
                skipped += 1
                continue

            node_label, edge, edge_weight = build_graph_from_calls(api_calls, edge_mode)
            sample_id = resolve_sample_id(raw_sample_id, row_index, id_counter, id_policy)
            graph = {
                "id": sample_id,
                "graph_label": graph_label,
                "number_node": len(node_label),
                "node_label": node_label,
                "number_edge": len(edge),
                "edge": edge,
                "edge_weight": edge_weight,
            }
            validate_graph(graph, row_index=row_index)
            graphs.append(graph)
            label_counter[graph_label] += 1

    duplicate_groups = sum(1 for _, cnt in id_counter.items() if cnt > 1)
    duplicate_total = sum(cnt - 1 for _, cnt in id_counter.items() if cnt > 1)
    return graphs, label_counter, skipped, duplicate_groups, duplicate_total


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_pkl = Path(args.output_pkl)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    graphs, label_counter, skipped, duplicate_groups, duplicate_total = convert_csv(
        input_csv=input_csv,
        label_format=args.label_format,
        edge_mode=args.edge_mode,
        limit=args.limit,
        id_policy=args.id_policy,
    )
    if not graphs:
        raise RuntimeError("No valid samples were converted.")

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as fp:
        pickle.dump(graphs, fp, protocol=pickle.HIGHEST_PROTOCOL)

    num_nodes = [g["number_node"] for g in graphs]
    num_edges = [g["number_edge"] for g in graphs]
    print(f"Saved {len(graphs)} graphs to {output_pkl}")
    print(f"Labels: {dict(label_counter)}")
    print(
        "Nodes per graph (min/mean/max): "
        f"{min(num_nodes)}/{sum(num_nodes)/len(num_nodes):.2f}/{max(num_nodes)}"
    )
    print(
        "Edges per graph (min/mean/max): "
        f"{min(num_edges)}/{sum(num_edges)/len(num_edges):.2f}/{max(num_edges)}"
    )
    print(f"Skipped rows: {skipped}")
    print(f"Duplicate id groups in input: {duplicate_groups}, duplicate rows: {duplicate_total}")
    print(f"id-policy: {args.id_policy}")


if __name__ == "__main__":
    main()

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from hydra._internal.config_repository import (
    ConfigRepository,
    IConfigRepository,
)
from hydra._internal.config_search_path_impl import ConfigSearchPathImpl
from hydra._internal.defaults_graph import (
    DefaultsGraph,
    GraphEdge,
    GraphNode,
    build_defaults_graph,
    format_defaults_graph,
)
from hydra.core.plugins import Plugins

Plugins.instance()

DATA_PATH = "file://tests/defaults_list/graph_data"


def create_repo() -> IConfigRepository:
    csp = ConfigSearchPathImpl()
    csp.append(provider="main", path=DATA_PATH)
    csp.append("schema", "structured://")
    return ConfigRepository(config_search_path=csp)


def get_node(graph: DefaultsGraph, config_path: str, package: str) -> GraphNode:
    matches = [
        node
        for node in graph.nodes
        if node.config_path == config_path and node.package == package
    ]
    assert len(matches) == 1, f"expected exactly one node for {(config_path, package)}"
    return matches[0]


def get_edges(graph: DefaultsGraph, config_path: str, package: str) -> list[GraphEdge]:
    return get_node(graph, config_path, package).edges


def find_edge(edges: list[GraphEdge], target: str, package: str) -> GraphEdge:
    matches = [
        edge for edge in edges if edge.target == target and edge.package == package
    ]
    assert len(matches) == 1
    return matches[0]


def test_primary_graph_expands_all_options() -> None:
    graph = build_defaults_graph(create_repo(), "config", prepend_hydra=False)

    primary = get_node(graph, "config", "")
    assert primary.primary
    targets = {(edge.target, edge.package) for edge in primary.edges}
    # Every option of every referenced group is reachable, not just the
    # combination selected in the defaults list.
    assert ("group1/file1", "group1") in targets
    assert ("group1/file2", "group1") in targets
    assert ("group2/a", "group2") in targets
    assert ("group2/b", "group2") in targets
    # Package relocation of the same group gives a parallel subtree.
    assert ("group1/file1", "relocated") in targets
    assert ("group1/file2", "relocated") in targets
    assert ("pkg/relocated", "custom") in targets
    assert ("pkg/global", "") in targets


def test_selected_edges_are_marked() -> None:
    graph = build_defaults_graph(create_repo(), "config", prepend_hydra=False)
    edges = get_edges(graph, "config", "")

    selected = find_edge(edges, "group1/file1", "group1")
    assert selected.selected
    assert not selected.listed

    relocated = find_edge(edges, "group1/file2", "relocated")
    assert relocated.selected

    assert not find_edge(edges, "group1/file2", "group1").selected
    assert not find_edge(edges, "group2/a", "group2").selected


def test_package_relocation_is_preserved_per_node() -> None:
    graph = build_defaults_graph(create_repo(), "config", prepend_hydra=False)

    # The same config reached through different packages yields distinct
    # nodes so the graph is stable after package relocation.
    get_node(graph, "group1/file1", "group1")
    get_node(graph, "group1/file1", "relocated")

    # "# @package custom" relocates both the config and its relative group.
    relocated = get_node(graph, "pkg/relocated", "custom")
    inner = find_edge(relocated.edges, "pkg/group1/inner", "custom.group1")
    assert inner.selected
    get_node(graph, "pkg/group1/inner", "custom.group1")

    # "# @package _global_" maps to the empty (global) package.
    get_node(graph, "pkg/global", "")


def test_unresolved_choice_is_enumerated_but_not_an_error() -> None:
    graph = build_defaults_graph(create_repo(), "config", prepend_hydra=False)

    assert len(graph.unresolved_choices) == 1
    choice = graph.unresolved_choices[0]
    assert choice.parent == "config"
    assert choice.group == "group2"
    assert choice.available == ["a", "b"]
    assert graph.missing == []
    assert graph.cycles == []
    assert graph.conflicts == []


def test_override_directive_is_recorded() -> None:
    graph = build_defaults_graph(create_repo(), "config", prepend_hydra=False)
    node = get_node(graph, "group1/file2", "group1")
    assert node.overrides == ["group1/group2=b"]


def test_cycle_is_reported_without_recursing() -> None:
    graph = build_defaults_graph(create_repo(), "cycle/a", prepend_hydra=False)

    assert graph.cycles == [["cycle/a", "cycle/b", "cycle/a"]]
    back_edge = find_edge(get_edges(graph, "cycle/b", ""), "cycle/a", "")
    assert back_edge.cycle
    # The repeated config is not expanded a second time.
    assert [
        node.config_path for node in graph.nodes if node.config_path == "cycle/a"
    ] == ["cycle/a"]


def test_missing_config_default_is_reported() -> None:
    graph = build_defaults_graph(
        create_repo(), "missing_config_ref", prepend_hydra=False
    )
    assert len(graph.missing) == 1
    missing = graph.missing[0]
    assert missing.kind == "config"
    assert missing.parent == "missing_config_ref"
    assert missing.ref == "group1/does_not_exist"
    edge = get_edges(graph, "missing_config_ref", "")[0]
    assert edge.missing


def test_missing_selected_option_is_reported() -> None:
    graph = build_defaults_graph(create_repo(), "missing_option", prepend_hydra=False)
    assert len(graph.missing) == 1
    missing = graph.missing[0]
    assert missing.kind == "option"
    assert missing.ref == "group1/does_not_exist"
    assert missing.available == ["file1", "file2"]


def test_optional_missing_is_not_reported_as_error() -> None:
    graph = build_defaults_graph(create_repo(), "optional_missing", prepend_hydra=False)
    assert graph.missing == []
    edge = get_edges(graph, "optional_missing", "")[0]
    assert edge.missing and edge.optional


def test_options_list_expands_every_listed_item() -> None:
    graph = build_defaults_graph(create_repo(), "multi", prepend_hydra=False)
    edges = get_edges(graph, "multi", "")
    assert {edge.target for edge in edges} == {"group1/file1", "group1/file2"}
    assert all(edge.listed for edge in edges)

    graph = build_defaults_graph(create_repo(), "multi_missing", prepend_hydra=False)
    missing = graph.missing
    assert len(missing) == 1
    assert missing[0].kind == "option"
    assert missing[0].ref == "group1/nope"
    bad = find_edge(get_edges(graph, "multi_missing", ""), "group1/nope", "group1")
    assert bad.listed and bad.missing


def test_interpolated_choice_enumerates_options_dynamically() -> None:
    graph = build_defaults_graph(create_repo(), "interpolated", prepend_hydra=False)

    assert len(graph.interpolations) == 1
    interpolation = graph.interpolations[0]
    assert interpolation.parent == "interpolated"
    assert interpolation.group == "group1"
    assert interpolation.available == ["file1", "file2"]

    edges = get_edges(graph, "interpolated", "")
    assert {edge.target for edge in edges} == {"group1/file1", "group1/file2"}
    assert all(edge.dynamic for edge in edges)
    # The dynamic options are still traversed.
    get_node(graph, "group1/file1", "group1")
    get_node(graph, "group1/file2", "group1")


def test_cross_source_same_name_conflict_is_reported() -> None:
    # Both sources expose the same relative config path; the first source
    # wins during composition.
    csp = ConfigSearchPathImpl()
    csp.append(provider="main", path=f"{DATA_PATH}/source_a")
    csp.append(provider="plugin", path=f"{DATA_PATH}/source_b")
    csp.append("schema", "structured://")
    repo = ConfigRepository(config_search_path=csp)

    graph = build_defaults_graph(repo, "shared", prepend_hydra=False)

    assert len(graph.conflicts) == 1
    conflict = graph.conflicts[0]
    assert conflict.config_path == "shared"
    assert len(conflict.providers) == 2
    assert "main" in conflict.providers[0]
    assert "plugin" in conflict.providers[1]


def test_formatted_output_contains_graph_and_summary() -> None:
    graph = build_defaults_graph(create_repo(), "missing_option", prepend_hydra=False)
    text = format_defaults_graph(graph)
    assert "missing_option (primary)" in text
    assert "group1=does_not_exist" in text
    assert "Defaults Graph Summary" in text
    assert "Missing configs (1)" in text


def test_formatted_output_reports_clean_summary() -> None:
    graph = build_defaults_graph(create_repo(), "multi", prepend_hydra=False)
    text = format_defaults_graph(graph)
    assert "No cycles, missing configs, or cross-source conflicts detected." in text

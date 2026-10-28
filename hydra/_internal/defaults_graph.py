# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Static defaults graph.

Unlike the Defaults Tree (see ``defaults_list.py``), which follows the single
combination selected for a composition, the defaults graph expands *every*
option of every reachable config group. It is used by ``--info defaults-graph``
to give maintainers a complete view of the option relationships reachable from
the primary config, along with diagnostics for cycles, missing configs and
config paths shadowed across multiple config sources.
"""

import copy
from dataclasses import dataclass, field

from hydra._internal.config_repository import IConfigRepository
from hydra.core.default_element import ConfigDefault, GroupDefault
from hydra.core.object_type import ObjectType

from .defaults_list import update_package_header

_GLOBAL_PACKAGE = "<global>"


@dataclass
class GraphEdge:
    # "config" for a ConfigDefault, "group" for an option of a GroupDefault
    kind: str
    # Resolved config path of the target, e.g. "group1/file1"
    target: str
    # Effective package the target content is merged into (after @package and
    # "# @package" relocation). Empty string means the global package.
    package: str
    # Absolute config group path for "group" edges
    group: str | None = None
    # Option name for "group" edges
    name: str | None = None
    # The option statically selected by the defaults list
    selected: bool = False
    # The option is an item of an options list (multi-select)
    listed: bool = False
    # The reference is marked "optional"
    optional: bool = False
    # The option is reachable via an interpolated choice that cannot be
    # resolved statically
    dynamic: bool = False
    # The target config does not exist in any source
    missing: bool = False
    # The edge closes a defaults cycle back to a config on the DFS stack
    cycle: bool = False


@dataclass
class GraphNode:
    # Config path of this node, e.g. "config", "group1/file1"
    config_path: str
    # Effective package this config instance is merged into. The same config
    # path may legitimately appear at several packages; each (path, package)
    # is a distinct node.
    package: str
    primary: bool = False
    edges: list[GraphEdge] = field(default_factory=list)
    # Defaults-list override directives ("key=name") declared in this config.
    # Overrides do not pull in a config file, so they are not edges.
    overrides: list[str] = field(default_factory=list)
    # Deduplication key for edges; not part of the graph's logical content.
    _edge_keys: set[tuple[object, ...]] = field(
        default_factory=set, compare=False, repr=False
    )


@dataclass
class MissingReference:
    parent: str
    # "config": a ConfigDefault path that does not exist
    # "option": a selected or listed group option that does not exist
    kind: str
    ref: str
    group: str | None = None
    available: list[str] = field(default_factory=list)


@dataclass
class UnresolvedChoice:
    parent: str
    group: str
    available: list[str] = field(default_factory=list)


@dataclass
class UnresolvedInterpolation:
    parent: str
    group: str | None
    raw: str
    available: list[str] = field(default_factory=list)


@dataclass
class SourceConflict:
    config_path: str
    # repr() of every source providing this path, in search path order.
    # The first entry is the source that wins during composition.
    providers: list[str]


@dataclass
class DefaultsGraph:
    root_config: str | None
    roots: list[str]
    nodes: list[GraphNode]
    cycles: list[list[str]]
    missing: list[MissingReference]
    unresolved_choices: list[UnresolvedChoice]
    interpolations: list[UnresolvedInterpolation]
    conflicts: list[SourceConflict]


class _GraphBuilder:
    def __init__(self, repo: IConfigRepository) -> None:
        self.repo = repo
        self.nodes: list[GraphNode] = []
        self._node_keys: set[tuple[str, str]] = set()
        self._stack: list[str] = []
        self.cycles: list[list[str]] = []
        self._cycle_keys: set[frozenset[tuple[str, str]]] = set()
        self.missing: list[MissingReference] = []
        self._missing_keys: set[tuple[str, str, str]] = set()
        self.unresolved_choices: list[UnresolvedChoice] = []
        self._choice_keys: set[tuple[str, str]] = set()
        self.interpolations: list[UnresolvedInterpolation] = []
        self._interpolation_keys: set[tuple[str, str | None, str]] = set()
        self.conflicts: dict[str, SourceConflict] = {}

    def build(self, config_name: str | None, prepend_hydra: bool) -> DefaultsGraph:
        roots: list[str] = []
        if config_name is not None:
            self._expand(config_name, package="", primary=True)
            roots.append(config_name)
        if prepend_hydra:
            self._expand("hydra/config", package="")
            roots.append("hydra/config")
        return DefaultsGraph(
            root_config=config_name,
            roots=roots,
            nodes=self.nodes,
            cycles=self.cycles,
            missing=self.missing,
            unresolved_choices=self.unresolved_choices,
            interpolations=self.interpolations,
            conflicts=list(self.conflicts.values()),
        )

    def _expand(
        self, path: str, package: str, primary: bool = False
    ) -> GraphNode | None:
        if path in self._stack:
            self._record_cycle(path)
            return None

        key = (path, package)
        if key in self._node_keys:
            return None
        self._node_keys.add(key)

        self._record_conflict(path)
        node = GraphNode(config_path=path, package=package, primary=primary)
        self.nodes.append(node)

        self._stack.append(path)
        try:
            loaded = self.repo.load_config(config_path=path)
            if loaded is None or loaded.defaults_list is None:
                return node
            parent_group = path.rpartition("/")[0]
            for raw in loaded.defaults_list:
                if raw.is_self():
                    continue
                element = copy.deepcopy(raw)
                element.update_parent(parent_group, package)
                if isinstance(element, GroupDefault) and element.is_override():
                    self._record_override(node, element)
                elif isinstance(element, ConfigDefault):
                    self._add_config_edge(node, element)
                else:
                    assert isinstance(element, GroupDefault)
                    self._add_group_edges(node, element)
        finally:
            self._stack.pop()
        return node

    def _record_override(self, node: GraphNode, element: GroupDefault) -> None:
        annotation = f"{element.get_override_key()}={element.get_name()}"
        if annotation not in node.overrides:
            node.overrides.append(annotation)

    def _add_edge(self, node: GraphNode, edge: GraphEdge) -> None:
        dedup_key = (edge.kind, edge.target, edge.package)
        if dedup_key in node._edge_keys:
            for existing in node.edges:
                if (
                    existing.kind == edge.kind
                    and existing.target == edge.target
                    and existing.package == edge.package
                ):
                    existing.selected = existing.selected or edge.selected
                    existing.dynamic = existing.dynamic or edge.dynamic
                    existing.listed = existing.listed or edge.listed
                    existing.missing = existing.missing or edge.missing
                    existing.cycle = existing.cycle or edge.cycle
                    return
        node._edge_keys.add(dedup_key)
        node.edges.append(edge)

    def _add_config_edge(self, node: GraphNode, element: ConfigDefault) -> None:
        target = element.get_config_path()
        edge = GraphEdge(
            kind="config",
            target=target,
            package=element.get_final_package(),
            optional=element.is_optional(),
        )
        if element.is_interpolation():
            # The referenced config path depends on runtime choices and
            # cannot be resolved statically.
            edge.dynamic = True
            self._add_interpolation(node.config_path, None, target, [])
            self._add_edge(node, edge)
            return
        exists = self.repo.config_exists(target)
        if exists:
            update_package_header(self.repo, element)
            edge.package = element.get_final_package()
        if not exists:
            edge.missing = True
            if not edge.optional:
                self._add_missing(
                    MissingReference(parent=node.config_path, kind="config", ref=target)
                )
            self._add_edge(node, edge)
            return
        edge.cycle = target in self._stack
        self._add_edge(node, edge)
        self._expand(target, edge.package)

    def _add_group_edges(self, node: GraphNode, element: GroupDefault) -> None:
        group = element.get_group_path()
        optional = element.is_optional()
        if self.repo.group_exists(group):
            options = self.repo.get_group_options(group, ObjectType.CONFIG)
        else:
            options = []

        def make_edge(
            name: str,
            *,
            exists: bool,
            selected: bool = False,
            listed: bool = False,
            dynamic: bool = False,
        ) -> GraphEdge:
            target = f"{group}/{name}"
            alt = copy.deepcopy(element)
            edge = GraphEdge(
                kind="group",
                target=target,
                package="",
                group=group,
                name=name,
                selected=selected,
                listed=listed,
                optional=optional,
                dynamic=dynamic,
            )
            if exists:
                alt.value = name
                update_package_header(self.repo, alt)
            edge.package = alt.get_final_package()
            return edge

        def visit(edge: GraphEdge) -> None:
            edge.cycle = edge.target in self._stack
            self._add_edge(node, edge)
            if not edge.missing and not edge.cycle:
                self._expand(edge.target, edge.package)

        if element.is_options():
            for name in element.get_options():
                exists = name in options
                edge = make_edge(name, exists=exists, listed=True)
                if not exists:
                    edge.missing = True
                    if not optional:
                        self._add_missing(
                            MissingReference(
                                parent=node.config_path,
                                kind="option",
                                ref=f"{group}/{name}",
                                group=group,
                                available=options,
                            )
                        )
                visit(edge)
            return

        name = element.get_name()
        if element.is_interpolation():
            assert name is not None
            self._add_interpolation(node.config_path, group, name, options)
            for option in options:
                visit(make_edge(option, exists=True, dynamic=True))
            return

        if name is None or element.is_missing():
            self._add_unresolved_choice(node.config_path, group, options)
            for option in options:
                visit(make_edge(option, exists=True))
            return

        assert name is not None
        selected_exists = name in options
        selected_edge = make_edge(name, exists=selected_exists, selected=True)
        if not selected_exists:
            selected_edge.missing = True
            if not optional:
                self._add_missing(
                    MissingReference(
                        parent=node.config_path,
                        kind="option",
                        ref=f"{group}/{name}",
                        group=group,
                        available=options,
                    )
                )
            visit(selected_edge)
            return

        visit(selected_edge)
        for option in options:
            if option == name:
                continue
            visit(make_edge(option, exists=True))

    def _add_missing(self, missing: MissingReference) -> None:
        key = (missing.parent, missing.kind, missing.ref)
        if key not in self._missing_keys:
            self._missing_keys.add(key)
            self.missing.append(missing)

    def _add_unresolved_choice(
        self, parent: str, group: str, options: list[str]
    ) -> None:
        key = (parent, group)
        if key not in self._choice_keys:
            self._choice_keys.add(key)
            self.unresolved_choices.append(
                UnresolvedChoice(parent=parent, group=group, available=options)
            )

    def _add_interpolation(
        self, parent: str, group: str | None, raw: str, options: list[str]
    ) -> None:
        key = (parent, group, raw)
        if key not in self._interpolation_keys:
            self._interpolation_keys.add(key)
            self.interpolations.append(
                UnresolvedInterpolation(
                    parent=parent, group=group, raw=raw, available=options
                )
            )

    def _record_conflict(self, path: str) -> None:
        if path in self.conflicts:
            return
        providers = [
            repr(source) for source in self.repo.get_sources() if source.is_config(path)
        ]
        if len(providers) > 1:
            self.conflicts[path] = SourceConflict(config_path=path, providers=providers)

    def _record_cycle(self, repeated: str) -> None:
        start = self._stack.index(repeated)
        sequence = self._stack[start:] + [repeated]
        cycle_key = frozenset(zip(sequence, sequence[1:]))
        if cycle_key not in self._cycle_keys:
            self._cycle_keys.add(cycle_key)
            self.cycles.append(sequence)


def build_defaults_graph(
    repo: IConfigRepository,
    config_name: str | None,
    prepend_hydra: bool = True,
) -> DefaultsGraph:
    """
    Walk every config reachable from the primary config and expand every option
    of every referenced config group.

    The walk is static: command-line group overrides are not applied, so all
    options remain visible. Config search path processing (including
    ``hydra.searchpath``) must already be reflected in ``repo``.
    """
    return _GraphBuilder(repo).build(config_name, prepend_hydra)


def format_defaults_graph(graph: DefaultsGraph) -> str:
    lines: list[str] = []

    for node in graph.nodes:
        if node.primary:
            title = f"{node.config_path} (primary)"
        elif node.config_path in graph.roots:
            title = f"{node.config_path} (hydra)"
        else:
            title = node.config_path
        lines.append(title)
        for annotation in node.overrides:
            lines.append(f"  override {annotation}")
        for edge in node.edges:
            lines.append("  " + _format_edge(edge))
        lines.append("")

    lines.extend(_format_summary(graph))
    return "\n".join(lines)


def _format_edge(edge: GraphEdge) -> str:
    package = edge.package if edge.package != "" else _GLOBAL_PACKAGE
    details = [f"package={package}"]
    if edge.kind == "group":
        details.insert(0, f"{edge.group}={edge.name}")
    for flag in (
        "selected",
        "listed",
        "dynamic",
        "optional",
        "missing",
        "cycle",
    ):
        if getattr(edge, flag):
            details.append(flag)
    return f"{edge.target} [{', '.join(details)}]"


def _format_summary(graph: DefaultsGraph) -> list[str]:
    lines = ["Defaults Graph Summary", "========================", ""]
    has_issues = False

    if graph.cycles:
        has_issues = True
        lines.append(f"Cycles ({len(graph.cycles)}):")
        for cycle in graph.cycles:
            lines.append("  " + " -> ".join(cycle))
        lines.append("")

    if graph.missing:
        has_issues = True
        lines.append(f"Missing configs ({len(graph.missing)}):")
        for missing in graph.missing:
            lines.append("  " + _format_missing(missing))
        lines.append("")

    if graph.conflicts:
        has_issues = True
        lines.append(f"Cross-source conflicts ({len(graph.conflicts)}):")
        for conflict in graph.conflicts:
            lines.append(
                f"  '{conflict.config_path}' is provided by "
                f"{len(conflict.providers)} sources (first wins):"
            )
            for index, provider in enumerate(conflict.providers):
                role = "used" if index == 0 else "shadowed"
                lines.append(f"    [{role}] {provider}")
        lines.append("")

    if graph.unresolved_choices:
        lines.append(
            f"Unresolved choices ({len(graph.unresolved_choices)}); "
            "select one at compose time:"
        )
        for choice in graph.unresolved_choices:
            options = (
                ", ".join(choice.available) if choice.available else "(none available)"
            )
            lines.append(
                f"  '{choice.group}' referenced from '{choice.parent}'; "
                f"available: {options}"
            )
        lines.append("")

    if graph.interpolations:
        lines.append(f"Unresolved interpolations ({len(graph.interpolations)}):")
        for interpolation in graph.interpolations:
            if interpolation.available:
                options = ", ".join(interpolation.available)
            else:
                options = "(cannot enumerate statically)"
            target = (
                f"group '{interpolation.group}'"
                if interpolation.group is not None
                else "config path"
            )
            lines.append(
                f"  '{interpolation.raw}' in '{interpolation.parent}' "
                f"({target}); possible options: {options}"
            )
        lines.append("")

    if not has_issues:
        lines.append("No cycles, missing configs, or cross-source conflicts detected.")
        lines.append("")

    return lines


def _format_missing(missing: MissingReference) -> str:
    prefix = f"[{missing.kind}] referenced from '{missing.parent}': "
    if missing.kind == "config":
        return prefix + f"config '{missing.ref}' not found"
    assert missing.kind == "option"
    options = ", ".join(missing.available) if missing.available else "(none available)"
    return prefix + f"selected option '{missing.ref}' not found; available: {options}"

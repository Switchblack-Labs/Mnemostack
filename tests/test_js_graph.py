"""JS/TS call-graph construction: nodes, same-file calls, cross-file links."""

from __future__ import annotations

import json

from mnemostack.core.retrieval.call_graph import CallGraph, EdgeType
from mnemostack.core.retrieval.js_graph import (
    build_nodes_for_js_file,
    link_js_file_imports,
)


def _build(graph, files):
    for f in files:
        build_nodes_for_js_file(f, graph=graph)
    for f in files:
        link_js_file_imports(f, graph=graph)


class TestJsGraph:
    def test_same_file_call_and_contains(self, tmp_path):
        graph = CallGraph(store_dir=tmp_path / "store")
        f = tmp_path / "app.js"
        f.write_text(
            "function helper() { return 1; }\n"
            "export function run() { return helper(); }\n"
        )
        _build(graph, [f])

        contains = graph.get_neighbors(
            str(f), hops=1, direction="outgoing", edge_types=(EdgeType.CONTAINS,)
        )
        assert f"{f}::helper" in contains and f"{f}::run" in contains
        calls = graph.get_neighbors(
            f"{f}::run", hops=1, direction="outgoing", edge_types=(EdgeType.CALLS,)
        )
        assert f"{f}::helper" in calls
        graph.close()

    def test_arrow_function_and_class_method(self, tmp_path):
        graph = CallGraph(store_dir=tmp_path / "store")
        f = tmp_path / "svc.ts"
        f.write_text(
            "const format = (x: string) => x.trim();\n"
            "export class Client {\n"
            "  send(x: string) { return format(x); }\n"
            "}\n"
        )
        _build(graph, [f])

        assert graph.has_node(f"{f}::format")
        assert graph.has_node(f"{f}::Client.send")
        calls = graph.get_neighbors(
            f"{f}::Client.send", hops=1, direction="outgoing", edge_types=(EdgeType.CALLS,)
        )
        assert f"{f}::format" in calls, "method call to a module-level arrow not linked"
        graph.close()

    def test_relative_import_links_across_files(self, tmp_path):
        graph = CallGraph(store_dir=tmp_path / "store")
        util = tmp_path / "util.ts"
        main = tmp_path / "main.ts"
        util.write_text("export function helper() { return 1; }\n")
        # TS ESM style: the specifier says .js but the file on disk is .ts
        main.write_text(
            "import { helper } from './util.js';\n"
            "export function run() { return helper(); }\n"
        )
        _build(graph, [util, main])

        imports = graph.get_neighbors(
            str(main), hops=1, direction="outgoing", edge_types=(EdgeType.IMPORTS_FROM,)
        )
        assert str(util) in imports
        calls = graph.get_neighbors(
            f"{main}::run", hops=1, direction="outgoing", edge_types=(EdgeType.CALLS,)
        )
        assert f"{util}::helper" in calls, "cross-file CALLS edge not created"
        graph.close()

    def test_directory_index_and_require(self, tmp_path):
        graph = CallGraph(store_dir=tmp_path / "store")
        pkg = tmp_path / "lib"
        pkg.mkdir()
        index = pkg / "index.js"
        index.write_text("function boot() { return 2; }\nmodule.exports = { boot };\n")
        main = tmp_path / "main.js"
        main.write_text("const lib = require('./lib');\n")
        _build(graph, [index, main])

        imports = graph.get_neighbors(
            str(main), hops=1, direction="outgoing", edge_types=(EdgeType.IMPORTS_FROM,)
        )
        assert str(index) in imports, "require of a directory index not resolved"
        graph.close()

    def test_installed_package_is_a_boundary(self, tmp_path):
        graph = CallGraph(store_dir=tmp_path / "store")
        dep = tmp_path / "node_modules" / "left-pad"
        dep.mkdir(parents=True)
        (dep / "package.json").write_text(json.dumps({"main": "./lib/pad.js"}))
        (dep / "lib").mkdir()
        entry = dep / "lib" / "pad.js"
        entry.write_text("module.exports = function pad() {};\n")
        main = tmp_path / "main.js"
        main.write_text("import pad from 'left-pad';\nexport function run() { return pad(); }\n")
        _build(graph, [main])

        assert graph.external_imports(str(main)) == [str(entry)], (
            "installed package boundary not recorded at its entry file"
        )
        graph.close()

    def test_unresolvable_import_stays_quiet(self, tmp_path):
        graph = CallGraph(store_dir=tmp_path / "store")
        main = tmp_path / "main.ts"
        main.write_text("import { x } from '@/aliased/thing';\nexport const run = () => x();\n")
        _build(graph, [main])

        assert graph.get_neighbors(
            str(main), hops=1, direction="outgoing", edge_types=(EdgeType.IMPORTS_FROM,)
        ) == []
        graph.close()

    def test_jsx_element_links_to_the_imported_component(self, tmp_path):
        graph = CallGraph(store_dir=tmp_path / "store")
        widget = tmp_path / "Widget.jsx"
        page = tmp_path / "Page.jsx"
        widget.write_text("export function Widget() { return <div />; }\n")
        page.write_text(
            "import { Widget } from './Widget';\n"
            "export function Page() { return <div><Widget /></div>; }\n"
        )
        _build(graph, [widget, page])

        calls = graph.get_neighbors(
            f"{page}::Page", hops=1, direction="outgoing", edge_types=(EdgeType.CALLS,)
        )
        assert f"{widget}::Widget" in calls, "rendered component not linked"
        graph.close()

    def test_default_import_and_namespace_call(self, tmp_path):
        graph = CallGraph(store_dir=tmp_path / "store")
        layout = tmp_path / "Layout.jsx"
        utils = tmp_path / "utils.js"
        page = tmp_path / "Page.jsx"
        layout.write_text("export default function Layout() { return <div />; }\n")
        utils.write_text("export function slugify(s) { return s; }\n")
        page.write_text(
            "import Layout from './Layout.jsx';\n"
            "import * as utils from './utils';\n"
            "export function Page() { return <Layout>{utils.slugify('x')}</Layout>; }\n"
        )
        _build(graph, [layout, utils, page])

        calls = graph.get_neighbors(
            f"{page}::Page", hops=1, direction="outgoing", edge_types=(EdgeType.CALLS,)
        )
        assert f"{layout}::Layout" in calls, "default-imported component not linked"
        assert f"{utils}::slugify" in calls, "namespace call not linked"
        graph.close()

    def test_default_import_renamed_to_an_undefined_symbol_makes_no_edge(self, tmp_path):
        # The importer can call a default export anything; guessing would be a lie.
        graph = CallGraph(store_dir=tmp_path / "store")
        mod = tmp_path / "mod.jsx"
        page = tmp_path / "Page.jsx"
        mod.write_text("export default function RealName() { return <div />; }\n")
        page.write_text(
            "import Whatever from './mod.jsx';\n"
            "export function Page() { return <Whatever />; }\n"
        )
        _build(graph, [mod, page])

        assert graph.get_neighbors(
            f"{page}::Page", hops=1, direction="outgoing", edge_types=(EdgeType.CALLS,)
        ) == []
        graph.close()

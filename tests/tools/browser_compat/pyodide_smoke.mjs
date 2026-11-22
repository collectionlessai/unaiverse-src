#!/usr/bin/env node
/**
 * Browser compatibility smoke test of a pure unaiverse wheel (plan, Phase 0.7).
 *
 * Runs Pyodide in Node, the way the JS SDK runs it in the browser: the SDK's own
 * mock setup code (PYTHON_SETUP_CODE, read from its Pyodide.js so the two never
 * drift apart), then Pillow and the given unaiverse wheel. It imports every
 * module the browser runtime loads and writes the API surface of those modules
 * (public and private callables with their signatures) as JSON, for
 * api_surface_diff.py.
 *
 * Usage (Node >= 18):
 *   node tests/tools/browser_compat/pyodide_smoke.mjs <wheel.whl> <out.json> [--js-root <dir>]
 *
 * --js-root is the folder holding node_modules/pyodide and unaiverse-js-src
 * (default: ../unaiverse-js, next to unaiverse-src). Exit code 0 when every
 * module imports, 1 otherwise. No network: the wheels are unpacked into
 * site-packages directly (micropip is not needed for local files).
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// Modules the browser runtime imports (from the SDK sources) plus the ones they pull in and run there
const MODULES = [
	'unaiverse.agent', 'unaiverse.agent_basics', 'unaiverse.interaction', 'unaiverse.hsm.hsm',
	'unaiverse.modules.utils', 'unaiverse.streams', 'unaiverse.clock', 'unaiverse.custom',
	'unaiverse.utils.misc', 'unaiverse.utils.logger', 'unaiverse.stats',
	'unaiverse.networking.node.connpool', 'unaiverse.networking.p2p.messages',
	'unaiverse.networking.session', 'unaiverse.networking.session.sess_msg',
	'unaiverse.networking.session.session', 'unaiverse.networking.session.session_manager',
];

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const jsRootFlag = args.indexOf('--js-root');
const jsRoot = path.resolve(jsRootFlag >= 0 ? args.splice(jsRootFlag, 2)[1]
	: path.join(here, '..', '..', '..', '..', 'unaiverse-js'));
const [wheel, outPath] = args;
if (!wheel || !outPath) {
	console.error('usage: pyodide_smoke.mjs <wheel.whl> <out.json> [--js-root <dir>]');
	process.exit(2);
}

const sdk = path.join(jsRoot, 'unaiverse-js-src');
const setupSrc = fs.readFileSync(path.join(sdk, 'src/unaiverse/Pyodide/Pyodide.js'), 'utf8');
const setupMatch = setupSrc.match(/const PYTHON_SETUP_CODE = `([\s\S]*?)`;/);
if (!setupMatch) {
	console.error('PYTHON_SETUP_CODE not found in the SDK Pyodide.js');
	process.exit(2);
}
const pillow = fs.readdirSync(path.join(sdk, 'public')).find((f) => f.startsWith('pillow-') && f.endsWith('.whl'));
const micropip = fs.readdirSync(path.join(sdk, 'public')).find((f) => f.startsWith('micropip-') && f.endsWith('.whl'));

const { loadPyodide } = await import(pathToFileURL(path.join(jsRoot, 'node_modules/pyodide/pyodide.mjs')).href);
const pyodide = await loadPyodide({ stdout: () => {}, stderr: () => {} });
console.log(`Pyodide ${pyodide.version}, Python ${pyodide.runPython('import sys; sys.version.split()[0]')}`);

// The SDK setup imports micropip first: unpack it, Pillow and the wheel under test into site-packages
const wheels = [path.join(sdk, 'public', micropip), path.join(sdk, 'public', pillow), path.resolve(wheel)];
wheels.forEach((w, i) => pyodide.FS.writeFile(`/tmp/w${i}.whl`, fs.readFileSync(w)));
pyodide.runPython(`
import site, zipfile
for i in range(${wheels.length}):
    zipfile.ZipFile(f"/tmp/w{i}.whl").extractall(site.getsitepackages()[0])
`);
console.log(`installed: ${wheels.map((w) => path.basename(w)).join(', ')}`);

pyodide.globals.set('_setup_code', setupMatch[1]);
pyodide.globals.set('_modules', pyodide.toPy(MODULES));
await pyodide.runPythonAsync('import textwrap\nexec(textwrap.dedent(_setup_code), {})');

const report = pyodide.runPython(`
import json, inspect, importlib, traceback

def _sig(obj):

    # Call shape only (names, kinds, defaults): annotations do not matter to a caller
    try:
        s = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    params = [p.replace(annotation=inspect.Parameter.empty) for p in s.parameters.values()]
    return str(s.replace(parameters=params, return_annotation=inspect.Signature.empty))

def _surface(mod):
    out = {}
    for name, obj in vars(mod).items():
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        if inspect.isclass(obj):
            members = {}
            for k, v in vars(obj).items():
                f = v.__func__ if isinstance(v, (staticmethod, classmethod)) else v
                if isinstance(f, property):
                    members[k] = "property"
                elif callable(f):
                    members[k] = ("async " if inspect.iscoroutinefunction(f) else "") + (_sig(f) or "?")
            out[name] = {"kind": "class", "bases": [b.__name__ for b in obj.__bases__], "members": members}
        elif callable(obj):
            out[name] = {"kind": "function", "sig": ("async " if inspect.iscoroutinefunction(obj) else "") +
                         (_sig(obj) or "?")}
    return out

res = {"modules": {}, "errors": {}}
for m in _modules:
    try:
        res["modules"][m] = _surface(importlib.import_module(m))
    except BaseException:
        res["errors"][m] = traceback.format_exc(limit=3)
json.dumps(res, sort_keys=True)
`);
fs.writeFileSync(outPath, report);
const parsed = JSON.parse(report);
for (const [m, err] of Object.entries(parsed.errors)) {
	console.error(`IMPORT FAILED ${m}\n${err}`);
}
console.log(`imported ${Object.keys(parsed.modules).length}/${MODULES.length} modules, surface written to ${outPath}`);
process.exit(Object.keys(parsed.errors).length ? 1 : 0);

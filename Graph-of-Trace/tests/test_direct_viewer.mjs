// Run with Node >=22.18 (native TS stripping), from the GoT repository root.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {convertGoTToFlow, layoutWithDagre} from '../frontend/src/got-to-reactflow.ts';

const file = process.argv[2] ?? 'runs/nedo-rsi-direct/smoke-test/got.json';
const graph = JSON.parse(fs.readFileSync(file, 'utf8'));
const flow = convertGoTToFlow(graph);
assert.deepEqual(flow.nodes.map(n => n.data.label), ['Baseline', 'Ablation A', 'Ablation B', 'Compare A/B']);
assert.equal(flow.nodes.some(n => n.data.isPlaceholder), false);
assert.deepEqual(flow.edges.map(e => `${e.source}->${e.target}`).sort(),
  ['N001->N001', 'N001->N002', 'N001->N003', 'N002->N004', 'N003->N004']);
const layout = layoutWithDagre(flow.nodes, flow.edges);
assert.equal(layout.nodes.length, 4);
assert.ok(layout.nodes.every(n => Number.isFinite(n.position.x) && Number.isFinite(n.position.y)));
const positions = Object.fromEntries(layout.nodes.map(n => [n.id, n.position]));
assert.equal(positions.N002.y, positions.N003.y);
assert.ok(positions.N004.y > positions.N002.y);
console.log('Viewer smoke passed: real frontend converter + Dagre, 4 nodes, sibling branches and multi-parent comparison.');

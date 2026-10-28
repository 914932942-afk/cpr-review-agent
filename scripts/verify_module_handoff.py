"""Verify the frozen mainline handoff contract without starting CV or LLM work."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'data/mainline-delivery-v1/module-contracts.json'

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    data = json.loads(CONTRACT.read_text(encoding='utf-8'))
    assert data['schema_version'] == 'module-handoff-v1'
    baseline = data['baseline']
    assert baseline['graph_version'] == 'cpr-review-12-v0.9-cascade'
    assert baseline['default_llm_mode'] == 'disabled'
    assert baseline['new_training'] is False and baseline['new_manual_labels'] is False
    for rel, expected in data['protected_files'].items():
        path = ROOT / rel
        assert path.exists(), f'missing protected file: {rel}'
        assert sha256(path) == expected, f'protected hash changed: {rel}'
    modules = data['modules']
    assert [m['id'] for m in modules] == ['M1', 'M2', 'M3', 'M4', 'M5']
    for module in modules:
        assert module['owner'] and module['algorithm']
        assert module['input_schema'] and module['output_schema']
        assert module['verify'] and module['metrics'] and module['boundary']
        for rel in module['owned_files']:
            assert (ROOT / rel).exists(), f"missing {module['id']} file: {rel}"
    print(json.dumps({'passed': True, 'modules': len(modules), 'protected_files': len(data['protected_files']), 'llm_mode': baseline['default_llm_mode']}, ensure_ascii=False))

if __name__ == '__main__':
    main()

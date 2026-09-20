from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


_DATASET_ALIASES = {
    'federation': 'federation',
    'federated': 'federation',
    'fed': 'federation',
    'repair': 'repair',
    'model_repair': 'repair',
}

_DATASET_DEFAULTS = {
    'federation': {
        'domain': 'Model Driven Engineering',
        'root_feature': 'Model_Federation',
    },
    'repair': {
        'domain': 'Model Repair',
        'root_feature': 'Repair_Technique',
    },
}


def prompt_choice(title: str, options: Tuple[str, ...]) -> str:
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    while True:
        choice = input("Select option: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Try again.")


def load_key_file(path: Path) -> Optional[str]:
    try:
        if path.exists():
            return path.read_text(encoding='utf-8').strip()
    except Exception:
        return None
    return None


def normalize_dataset_name(dataset: str) -> str:
    key = str(dataset or '').strip().lower()
    if not key:
        return 'federation'
    return _DATASET_ALIASES.get(key, key)


def dataset_choices() -> Tuple[str, ...]:
    return tuple(_DATASET_DEFAULTS.keys())


def dataset_defaults(dataset: str) -> Dict[str, str]:
    normalized = normalize_dataset_name(dataset)
    if normalized not in _DATASET_DEFAULTS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {', '.join(dataset_choices())}")
    return dict(_DATASET_DEFAULTS[normalized])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def high_level_features_config_path(dataset: str) -> Path:
    normalized = normalize_dataset_name(dataset)
    return _repo_root() / 'config' / 'high_level_features' / f'{normalized}.yaml'


def load_high_level_features(*, dataset: str = 'federation', config_path: Optional[Path] = None) -> Dict[str, str]:
    path = config_path.expanduser().resolve() if config_path else high_level_features_config_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f'High-level feature config not found: {path}')
    if yaml is None:
        raise RuntimeError('PyYAML is required to load high-level feature config files')
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'High-level feature config must be a mapping: {path}')
    normalized = {}
    for key, value in data.items():
        k = str(key).strip()
        v = str(value).strip()
        if k and v:
            normalized[k] = v
    if not normalized:
        raise ValueError(f'No high-level features found in config: {path}')
    return normalized


def default_high_level_features(dataset: str = 'federation'):
    return load_high_level_features(dataset=dataset)

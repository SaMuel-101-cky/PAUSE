from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class GPFoldIdentity:
    canonical_fold_id: int
    split_fold_id: int
    subject_id: int | None

    @property
    def fold_stem(self) -> str:
        return f"fold{self.canonical_fold_id}"

    @property
    def subject_stem(self) -> str | None:
        if self.subject_id is None:
            return None
        return f"subject{self.subject_id}"


def identity_for_fold(dataset: str, fold) -> GPFoldIdentity:
    split_id = int(fold.fold_id)
    if dataset == "dreams":
        if not 1 <= split_id <= 20:
            raise ValueError(
                f"DREAMS split fold id must be in [1, 20], got {split_id}"
            )
        return GPFoldIdentity(
            canonical_fold_id=split_id - 1,
            split_fold_id=split_id,
            subject_id=split_id,
        )
    if not 0 <= split_id < 20:
        raise ValueError(f"{dataset} fold id must be in [0, 19], got {split_id}")
    return GPFoldIdentity(
        canonical_fold_id=split_id,
        split_fold_id=split_id,
        subject_id=None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_read_artifact(
    directory: str | Path,
    identity: GPFoldIdentity,
    suffix: str,
) -> Path:
    directory = Path(directory)
    canonical = directory / f"{identity.fold_stem}{suffix}"
    legacy = (
        None
        if identity.subject_stem is None
        else directory / f"{identity.subject_stem}{suffix}"
    )

    if canonical.is_file() and legacy is not None and legacy.is_file():
        if _sha256(canonical) != _sha256(legacy):
            raise ValueError(
                f"conflicting DREAMS artifacts: {canonical} and {legacy}"
            )
        return canonical
    if canonical.is_file():
        return canonical
    if legacy is not None and legacy.is_file():
        return legacy

    candidates = [str(canonical)]
    if legacy is not None:
        candidates.append(str(legacy))
    raise FileNotFoundError("missing artifact; tried " + ", ".join(candidates))

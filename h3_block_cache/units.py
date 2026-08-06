from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class CacheUnit:
    start: int
    stop: int

    def __post_init__(self):
        if not (0 <= self.start <= self.stop < 50):
            raise ValueError(f"invalid H3 block unit {self.start}-{self.stop}")

    @property
    def key(self):
        return f"{self.start}" if self.start == self.stop else f"{self.start}-{self.stop}"

    def contains(self, index):
        return self.start <= index <= self.stop


def parse_unit_spec(spec: str):
    units = []
    for raw in (spec or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" in raw:
            a, b = raw.split("-", 1)
            unit = CacheUnit(int(a), int(b))
        else:
            unit = CacheUnit(int(raw), int(raw))
        units.append(unit)
    if not units:
        raise ValueError("unit_spec must select at least one H3 block")
    units = sorted(units)
    for left, right in zip(units, units[1:]):
        if right.start <= left.stop:
            raise ValueError(f"overlapping cache units: {left.key} and {right.key}")
    return tuple(units)


def unit_map(units):
    out = {}
    for unit in units:
        for index in range(unit.start, unit.stop + 1):
            out[index] = unit
    return out

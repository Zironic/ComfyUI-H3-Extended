"""AIMDO page accounting for the H3 VRAM capacity guard."""

from dataclasses import dataclass


PAGE_SIZE = 32 * 1024 * 1024


@dataclass(frozen=True)
class WeightFootprint:
    resident_pages: int
    pinned_pages: int
    resident_unpinned_pages: int
    mandatory_pages: int
    mandatory_pinned_pages: int
    mandatory_group: str | None

    @property
    def resident_unpinned_bytes(self):
        return self.resident_unpinned_pages * PAGE_SIZE

    @property
    def mandatory_bytes(self):
        return (self.mandatory_pages - self.mandatory_pinned_pages) * PAGE_SIZE


def pages_for_allocation(allocation):
    """Return ``(vbar, page)`` keys touched by one ``module._v`` allocation."""
    vbar, address, size = allocation
    address = int(address)
    size = int(size)
    base = int(vbar.base_addr)
    if size < 0 or address < base:
        raise ValueError("invalid VBAR allocation")
    if size == 0:
        return frozenset()
    first = (address - base) // PAGE_SIZE
    stop = (address - base + size + PAGE_SIZE - 1) // PAGE_SIZE
    return frozenset((vbar, page) for page in range(first, stop))


def _matches_device(allocation, device):
    wanted = getattr(device, "index", None)
    have = getattr(allocation[0], "device", None)
    return wanted is None or have is None or int(have) == int(wanted)


def page_union(modules, device=None):
    pages = set()
    for module in modules:
        allocation = getattr(module, "_v", None)
        if allocation is not None and _matches_device(allocation, device):
            pages.update(pages_for_allocation(allocation))
    return frozenset(pages)


def _all_modules(module):
    if module is None:
        return ()
    modules = getattr(module, "modules", None)
    return tuple(modules()) if callable(modules) else (module,)


def _diffusion_model(model):
    return getattr(model, "diffusion_model", model)


def acquisition_groups(model, device=None):
    """Return the weight sets that can be required concurrently by H3.

    Dynamic prefetch and the shared compiled lease both acquire a complete main
    DiT block. Other phases are kept separate so fifty sequential blocks are
    never mistaken for one concurrent weight set.
    """
    model = _diffusion_model(model)
    groups = []

    for name in ("video_patch_proj", "audio_patch_proj", "condition_proj", "time_embedder"):
        module = getattr(model, name, None)
        if module is not None:
            groups.append(("prelude.%s" % name, page_union(_all_modules(module), device=device)))

    token_refiner = getattr(model, "token_refiner", None)
    for index, block in enumerate(getattr(token_refiner, "blocks", ())):
        groups.append(("token_refiner.%d" % index, page_union(_all_modules(block), device=device)))
    token_final = getattr(token_refiner, "final_norm", None)
    if token_final is not None:
        groups.append(("token_refiner.final_norm", page_union(_all_modules(token_final), device=device)))

    for index, block in enumerate(getattr(model, "blocks", ())):
        groups.append(("blocks.%d" % index, page_union(_all_modules(block), device=device)))

    final_layer = getattr(model, "final_layer", None)
    if final_layer is not None:
        groups.append(("final_layer", page_union(_all_modules(final_layer), device=device)))

    return tuple((name, pages) for name, pages in groups if pages)


def _inventory_pages(model, device=None):
    pages = set()
    for module in _all_modules(model):
        allocation = getattr(module, "_v", None)
        if allocation is None:
            continue
        if not _matches_device(allocation, device):
            continue
        pages.update(pages_for_allocation(allocation))
    return frozenset(pages)


def residency_flags(pages):
    """Read each referenced VBAR once and return flags keyed by page."""
    by_vbar = {}
    for vbar, page in pages:
        by_vbar.setdefault(vbar, set()).add(page)
    flags = {}
    for vbar, indices in by_vbar.items():
        residency = list(vbar.get_residency())
        for page in indices:
            if page < 0 or page >= len(residency):
                raise RuntimeError("VBAR residency does not cover allocated page %d" % page)
            value = int(residency[page])
            if value & 2 and not value & 1:
                raise RuntimeError("VBAR page %d is pinned but not resident" % page)
            flags[(vbar, page)] = value
    return flags


def footprint(model, device=None):
    """Measure reclaimable pages and the largest incremental H3 weight group."""
    inventory = _inventory_pages(model, device=device)
    flags = residency_flags(inventory)
    resident = {page for page, value in flags.items() if value & 1}
    pinned = {page for page, value in flags.items() if value & 2}
    resident_unpinned = resident - pinned

    groups = acquisition_groups(model, device=device)
    if not groups:
        return WeightFootprint(len(resident), len(pinned), len(resident_unpinned), 0, 0, None)

    group_name, mandatory = max(groups, key=lambda item: (len(item[1] - pinned), len(item[1])))
    mandatory_pinned = mandatory & pinned
    return WeightFootprint(
        len(resident),
        len(pinned),
        len(resident_unpinned),
        len(mandatory),
        len(mandatory_pinned),
        group_name,
    )

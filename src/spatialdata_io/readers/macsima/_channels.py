"""Plan one MACSima stack: channel selection, order, names, canvas, and nuclei split."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

    from spatialdata_io.readers.macsima._layout import StackFiles
    from spatialdata_io.readers.macsima._metadata import ChannelSource

_CONTEXT = ReaderErrorContext("macsima", "channel stack")

# Zero rows before and after, then zero columns before and after, of one channel.
type Padding = tuple[tuple[int, int], tuple[int, int]]

NO_PADDING: Padding = ((0, 0), (0, 0))


@dataclass(frozen=True, slots=True)
class ChannelOptions:
    """Validated caller choices for channel selection and naming."""

    exclude_cycles: frozenset[int]
    include_cycle_in_channel_names: bool
    nuclei_channel_name: str
    split_threshold_nuclei_channel: int | None


@dataclass(frozen=True, slots=True)
class PlannedChannel:
    """One selected channel, its unique stack name, and its YX zero padding."""

    source: ChannelSource
    name: str
    padding: Padding


@dataclass(frozen=True, slots=True)
class StackPlan:
    """Channels of the main stack and of the optional separate nuclei stack.

    Both stacks share one canvas, one pixel size, and the channel names
    assigned over the whole stack, so a channel has the same name in both.
    """

    name: str
    channels: tuple[PlannedChannel, ...]
    nuclei: tuple[PlannedChannel, ...] | None
    pixel_size_um: float


def make_channel_options(
    *,
    exclude_cycles: Sequence[int],
    include_cycle_in_channel_names: bool,
    nuclei_channel_name: str,
    split_threshold_nuclei_channel: int | None,
) -> ChannelOptions:
    """Validate caller choices once and own the borrowed cycle sequence."""
    if isinstance(exclude_cycles, str | bytes) or not isinstance(exclude_cycles, Sequence):
        message = "exclude_cycles must be a sequence of cycle numbers."
        raise TypeError(message)
    if any(isinstance(cycle, bool) or not isinstance(cycle, int) for cycle in exclude_cycles):
        message = "exclude_cycles must contain only integers."
        raise TypeError(message)
    if not isinstance(include_cycle_in_channel_names, bool):
        message = "include_cycle_in_channel_names must be a boolean."
        raise TypeError(message)

    if not isinstance(nuclei_channel_name, str):
        message = "nuclei_channel_name must be a string."
        raise TypeError(message)
    if not nuclei_channel_name:
        message = "nuclei_channel_name must not be empty."
        raise ValueError(message)

    threshold = split_threshold_nuclei_channel
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, int)):
        message = "split_threshold_nuclei_channel must be None or an integer."
        raise TypeError(message)
    if threshold is not None and threshold < 0:
        message = "split_threshold_nuclei_channel must not be negative."
        raise ValueError(message)

    return ChannelOptions(
        exclude_cycles=frozenset(exclude_cycles),
        include_cycle_in_channel_names=include_cycle_in_channel_names,
        nuclei_channel_name=nuclei_channel_name,
        split_threshold_nuclei_channel=threshold,
    )


def _require_single(values: AbstractSet[object], field: str, stack: StackFiles) -> None:
    """Reject a stack whose channels disagree on one acquisition property."""
    if len(values) > 1:
        raise ReaderFormatError(
            context=_CONTEXT,
            path=stack.directory,
            found=f"stack {stack.name!r} with {field} values {sorted(map(str, values))}",
            expected=f"one {field} per stack",
        )


def _require_consistent(sources: Sequence[ChannelSource], stack: StackFiles) -> None:
    """Stack only channels acquired on one grid of one ROI."""
    _require_single({source.dtype for source in sources}, "dtype", stack)
    _require_single({source.pixel_size_um for source in sources}, "pixel size (µm)", stack)
    _require_single({source.major for source in sources}, "MACSiQView major version", stack)
    _require_single({source.metadata.well for source in sources} - {None}, "well", stack)
    _require_single({source.metadata.roi for source in sources} - {None}, "ROI", stack)


def _names(sources: Sequence[ChannelSource], *, include_cycle: bool, stack: StackFiles) -> tuple[str, ...]:
    """Return unique names: repeats are numbered in stack order and collisions rejected."""
    bases = [
        f"R{source.metadata.cycle} {source.metadata.name}" if include_cycle else source.metadata.name
        for source in sources
    ]
    counts = Counter(bases)
    seen: Counter[str] = Counter()
    names: list[str] = []
    for base in bases:
        seen[base] += 1
        names.append(f"{base} ({seen[base]})" if counts[base] > 1 else base)

    # A numbered name can equal a real marker name, for example "DAPI (1)".
    if len(set(names)) != len(names):
        raise ReaderFormatError(
            context=_CONTEXT,
            path=stack.directory,
            found=f"channel names {names}",
            expected="unique names after numbering repeated markers",
        )
    return tuple(names)


def _place_on_canvas(sources: Sequence[ChannelSource]) -> tuple[tuple[Padding, ...], tuple[int, int]]:
    """Return each channel's zero padding onto one shared canvas, and the canvas shape.

    The canvas starts at the smallest plane position and spans every channel,
    so a position shared by all channels leaves no padding.
    """
    min_y = min(source.offset_yx[0] for source in sources)
    min_x = min(source.offset_yx[1] for source in sources)
    starts = [(source.offset_yx[0] - min_y, source.offset_yx[1] - min_x) for source in sources]
    canvas = (
        max(y + source.shape_yx[0] for (y, _), source in zip(starts, sources, strict=True)),
        max(x + source.shape_yx[1] for (_, x), source in zip(starts, sources, strict=True)),
    )

    paddings = tuple(
        ((y, canvas[0] - y - source.shape_yx[0]), (x, canvas[1] - x - source.shape_yx[1]))
        for (y, x), source in zip(starts, sources, strict=True)
    )
    return paddings, canvas


def plan_stack(stack: StackFiles, sources: Sequence[ChannelSource], options: ChannelOptions) -> StackPlan:
    """Select, order, name, and place the channels of one stack.

    Channels are ordered by cycle, then name, then file path. Excluded cycles
    are removed first. When more stain channels named exactly
    ``nuclei_channel_name`` exist than the split threshold, they form a
    separate nuclei stack; the main stack keeps the first and the last of them
    (the registration reference and the end-of-run image).
    """
    selected = sorted(
        (source for source in sources if source.metadata.cycle not in options.exclude_cycles),
        key=lambda source: (source.metadata.cycle, source.metadata.name, source.tiff.path.as_posix()),
    )
    if not selected:
        message = f"exclude_cycles removes every channel of stack {stack.name!r} in {str(stack.directory)!r}."
        raise ValueError(message)
    _require_consistent(selected, stack)

    names = _names(selected, include_cycle=options.include_cycle_in_channel_names, stack=stack)
    paddings, canvas = _place_on_canvas(selected)
    if any(padding != NO_PADDING for padding in paddings):
        warnings.warn(
            f"Zero-padding the channels of stack {stack.name!r} to the common canvas {canvas}.",
            UserWarning,
            stacklevel=3,
        )
    channels = tuple(
        PlannedChannel(source, name, padding) for source, name, padding in zip(selected, names, paddings, strict=True)
    )
    pixel_size_um = selected[0].pixel_size_um

    nuclei_indices = [
        index
        for index, source in enumerate(selected)
        if source.metadata.image_type == "stain" and source.metadata.name == options.nuclei_channel_name
    ]
    threshold = options.split_threshold_nuclei_channel
    if threshold is None or len(nuclei_indices) <= threshold:
        return StackPlan(name=stack.name, channels=channels, nuclei=None, pixel_size_um=pixel_size_um)

    moved = set(nuclei_indices[1:-1])
    return StackPlan(
        name=stack.name,
        channels=tuple(channel for index, channel in enumerate(channels) if index not in moved),
        nuclei=tuple(channels[index] for index in nuclei_indices),
        pixel_size_um=pixel_size_um,
    )

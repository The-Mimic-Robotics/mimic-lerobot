import re

CANONICAL_TEMPLATE = "Pick the red X, hand over, then place in cell <{CELL}> of the 3x3 board."

POSITION_ALIASES = {
    "top_left": {"top_left", "top left", "upper left", "upper_left", "topleft"},
    "top_middle": {"top_middle", "top middle", "upper middle", "upper_middle", "top center", "top_center", "topcentre", "top_centre"},
    "top_right": {"top_right", "top right", "upper right", "upper_right", "topright"},
    "middle_left": {"middle_left", "middle left", "center left", "centre left", "mid left", "middleleft"},
    "middle_middle": {"middle_middle", "middle middle", "center", "centre", "center middle", "middle center", "middle_center", "middlecentre", "middle_centre"},
    "middle_right": {"middle_right", "middle right", "center right", "centre right", "mid right", "middleright"},
    "bottom_left": {"bottom_left", "bottom left", "lower left", "bottomleft"},
    "bottom_middle": {"bottom_middle", "bottom middle", "lower middle", "bottom center", "bottom_center", "bottomcentre", "bottom_centre"},
    "bottom_right": {"bottom_right", "bottom right", "lower right", "bottomright"},
}

POSITION_TO_SLOT = {
    "top_left": "TOP_LEFT",
    "top_middle": "TOP_MIDDLE",
    "top_right": "TOP_RIGHT",
    "middle_left": "MIDDLE_LEFT",
    "middle_middle": "MIDDLE_MIDDLE",
    "middle_right": "MIDDLE_RIGHT",
    "bottom_left": "BOTTOM_LEFT",
    "bottom_middle": "BOTTOM_MIDDLE",
    "bottom_right": "BOTTOM_RIGHT",
}

LEGACY_TASK_TO_POSITION = {
    "pick red x handover place top left": "top_left",
    "pick red x handover place top middle": "top_middle",
    "pick red x handover place top right": "top_right",
    "pick red x handover place middle left": "middle_left",
    "pick red x handover place center": "middle_middle",
    "pick red x handover place middle right": "middle_right",
    "pick red x handover place bottom left": "bottom_left",
    "pick red x handover place bottom middle": "bottom_middle",
    "pick red x handover place bottom right": "bottom_right",
}


def normalize_position(position_text: str) -> str:
    txt = re.sub(r"\s+", " ", position_text.strip().lower().replace("-", " ").replace("_", " "))
    txt_underscore = txt.replace(" ", "_")
    for canonical, aliases in POSITION_ALIASES.items():
        if txt in aliases or txt_underscore in aliases:
            return canonical
    raise ValueError(f"Unrecognized position: {position_text}")


def canonical_instruction_from_position(position_text: str) -> str:
    position = normalize_position(position_text)
    return CANONICAL_TEMPLATE.format(CELL=POSITION_TO_SLOT[position])


def canonical_instruction_from_legacy_task(task_label: str) -> str:
    task_clean = re.sub(r"\s+", " ", task_label.strip().lower())
    if task_clean not in LEGACY_TASK_TO_POSITION:
        raise ValueError(f"Unsupported legacy task label: {task_label}")
    return canonical_instruction_from_position(LEGACY_TASK_TO_POSITION[task_clean])

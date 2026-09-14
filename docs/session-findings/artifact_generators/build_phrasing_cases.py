import json

CASES = {
    20: {"category": "table", "human_name": "table", "relation": "left",
         "object_center": [15.43130111694336, 0.36171889305114746, -2.6567201614379883],
         "start_position": [17.257476806640625, 0.17162801325321198, 0.5036007165908813],
         "start_rotation": [0.0, 0.25899821297535264, 0.0, 0.965877800591552],
         "target": [12.961536407470703, 0.17162801325321198, -1.2295787334442139],
         "orig_distance": 5.33},
    14: {"category": "sofa", "human_name": "sofa", "relation": "right",
         "object_center": [5.893670082092285, 1.1973400115966797, 2.3310203552246094],
         "start_position": [10.09335994720459, 0.26567015051841736, 2.3310203552246094],
         "start_rotation": [0.0, 0.7071067811865475, 0.0, 0.7071067811865476],
         "target": [5.901913642883301, 0.17162801325321198, -0.6812316179275513],
         "orig_distance": 8.72},
    4: {"category": "chest_of_drawers", "human_name": "dresser", "relation": "behind",
        "object_center": [16.3257999420166, 0.4626607894897461, -7.64518928527832],
        "start_position": [19.015472412109375, 0.17162801325321198, -7.580921649932861],
        "start_rotation": [0.0, 0.6986102613462049, 0.0, 0.7155024128133931],
        "target": [14.586729049682617, 0.17162801325321198, -7.306311130523682],
        "orig_distance": 5.13, "orig_steps": 210},
    13: {"category": "shelving", "human_name": "bookshelf", "relation": "next to",
         "object_center": [6.043270111083984, 0.9284069538116455, -3.1874396800994873],
         "start_position": [9.386150360107422, 0.17162801325321198, -2.8027701377868652],
         "start_rotation": [0.0, 0.6654634021498922, 0.0, 0.7464304792806166],
         "target": [6.214212417602539, 0.17162801325321198, -4.652770042419434],
         "orig_distance": 13.06},
}

PHRASING_TEMPLATES = {
    "left": {
        "bare": "Move to the left of the {name}.",
        "verb_stop": "Walk to the left of the {name} and stop there.",
        "two_step": "Walk toward the {name}. When you reach it, step to its left side and stop.",
        "anchored": "There is a {name} ahead of you. Walk up to the {name}, then move around to its left side and stop right next to it.",
        "narrated": "Walk forward toward the {name} in the room ahead. Once you reach the {name}, turn left and take a few steps to position yourself on the left side of it, then stop.",
    },
    "right": {
        "bare": "Move to the right of the {name}.",
        "verb_stop": "Walk to the right of the {name} and stop there.",
        "two_step": "Walk toward the {name}. When you reach it, step to its right side and stop.",
        "anchored": "There is a {name} ahead of you. Walk up to the {name}, then move around to its right side and stop right next to it.",
        "narrated": "Walk forward toward the {name} in the room ahead. Once you reach the {name}, turn right and take a few steps to position yourself on the right side of it, then stop.",
    },
    "behind": {
        "bare": "Go behind the {name}.",
        "verb_stop": "Walk behind the {name} and stop there.",
        "two_step": "Walk toward the {name}. When you reach it, go around to the far side of it and stop.",
        "anchored": "There is a {name} ahead of you. Walk up to the {name}, then continue past it to the far side, behind it, and stop.",
        "narrated": "Walk forward toward the {name} in the room ahead. Once you reach the {name}, keep going past it until you are on the opposite side from where you started, then stop.",
    },
    "next to": {
        "bare": "Stand next to the {name}.",
        "verb_stop": "Walk over to the {name} and stand right beside it.",
        "two_step": "Walk toward the {name}. When you reach it, stop right beside it.",
        "anchored": "There is a {name} ahead of you. Walk up to the {name} and stop as close to it as you can, right next to it.",
        "narrated": "Walk forward toward the {name} in the room ahead. Once you get close to the {name}, stop right beside it, touching distance away.",
    },
}


def main():
    out = []
    for ep_id, case in CASES.items():
        templates = PHRASING_TEMPLATES[case["relation"]]
        for tier, template in templates.items():
            instruction = template.format(name=case["human_name"])
            out.append({
                "base_episode_id": ep_id,
                "tier": tier,
                "instruction": instruction,
                **{k: v for k, v in case.items() if k not in ("orig_distance", "orig_steps")},
                "orig_distance": case.get("orig_distance"),
                "orig_steps": case.get("orig_steps"),
            })
    with open("/tmp/claude-1000/-home-storms-end-ETPNav/bf82bef2-3c40-490f-95a2-ed73feb09b45/scratchpad/phrasing_cases.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(out)} phrasing runs")


if __name__ == "__main__":
    main()

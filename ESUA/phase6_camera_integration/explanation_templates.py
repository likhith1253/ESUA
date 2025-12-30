# Templates for Human-Readable Explanations

# Structure:
# 1. Observation
# 2. General Principle
# 3. Possible Consequence
# 4. Gentle Suggestion

TEMPLATES = {
    'spill_risk': [
        "A {obj_a} is placed close to a {obj_b}.",
        "Liquids near electronic devices can be risky.",
        "If the liquid spills, it may damage the device.",
        "Moving the {obj_a} away could help reduce this risk."
    ],
    'fire_risk': [
        "A {obj_a} is currently near a {obj_b}.",
        "Heat sources placed near flammable objects can be dangerous.",
        "There is a potential risk of fire if they are left too close.",
        "It would be safer to separate the {obj_a} from the {obj_b}."
    ],
    'damage_risk': [
        "A {obj_a} is located near a {obj_b}.",
        "Liquids can easily damage paper-based items.",
        "If a spill occurs, the {obj_b} could be ruined.",
        "Please consider keeping the area around the {obj_b} clear."
    ],
    'sharp_risk': [
        "A {obj_a} was detected near the edge or near a {obj_b}.",
        "Sharp objects can cause injury if not stored safely.",
        "An accidental bump could cause the {obj_a} to fall or hurt someone.",
        "Storing the {obj_a} in a safer spot is recommended."
    ],
    'default': [
        "A {obj_a} is near a {obj_b}.",
        "Objects placed close together can sometimes interact unexpectedly.",
        "It is good practice to correct valid spatial organization.",
        "Please check if this arrangement is intended."
    ]
}

def get_explanation(risk_type, context):
    """
    Fills the template for the given risk type with context data.
    
    Args:
        risk_type (str): Key for the template (e.g., 'spill_risk').
        context (dict): Dictionary with values to fill (e.g., {'obj_a': 'cup'}).
        
    Returns:
        str: A multi-line string with the generated explanation.
    """
    if risk_type not in TEMPLATES:
        risk_type = 'default'
        
    lines = TEMPLATES[risk_type]
    
    # helper to format each line
    formatted_lines = []
    for line in lines:
        try:
            formatted_lines.append(line.format(**context))
        except KeyError as e:
            formatted_lines.append(line + f" [Missing data: {e}]")
            
    return "\n".join(formatted_lines)

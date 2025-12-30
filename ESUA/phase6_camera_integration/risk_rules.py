# Simple Rule Engine for Risk Detection

def check_risks(obj_a, obj_b, distance, relation_type):
    """
    Checks for risks between two objects based on their categories and proximity.
    
    Args:
        obj_a (dict): Object A details with 'name' and 'categories'.
        obj_b (dict): Object B details with 'name' and 'categories'.
        distance (float): Euclidean distance between centers.
        relation_type (str): 'near' or 'far from'.
        
    Returns:
        list: A list of detected risk strings.
    """
    risks = []
    
    cats_a = obj_a['categories']
    cats_b = obj_b['categories']
    
    # Rule 1: Spill Risk
    # IF (liquid) NEAR (electronics) -> Spill Risk
    is_liquid_a = 'liquid' in cats_a
    is_liquid_b = 'liquid' in cats_b
    is_elec_a = 'electronics' in cats_a
    is_elec_b = 'electronics' in cats_b
    
    if relation_type == "near":
        if (is_liquid_a and is_elec_b) or (is_liquid_b and is_elec_a):
            risks.append(f"Spill Risk detected: {obj_a['name']} is near {obj_b['name']}")

    # Rule 2: Fire Risk (Hypothetical for this demo as we might not have heat sources like 'candle' in COCO easily available in everyday office pics, but added for structure)
    # IF (heat) NEAR (flammable) -> Fire Risk
    # For demo purposes, let's assume 'laptop' (can get hot) near 'paper'/'book' is a low fire risk to show logic working if user wants.
    # Let's strictly stick to requested logic: heat source. (candle, toaster - not in typical office pic).
    # Let's add a placeholder rule.
    
    # Rule 3: Clutter/Organization (Example of non-safety rule)
    # IF (flammable/book) NEAR (liquid) -> Damage Risk
    is_flam_a = 'flammable' in cats_a
    is_flam_b = 'flammable' in cats_b
    
    if relation_type == "near":
        if (is_liquid_a and is_flam_b) or (is_liquid_b and is_flam_a):
             risks.append(f"Damage Risk detected: {obj_a['name']} (liquid) is near {obj_b['name']}")
             
    return risks

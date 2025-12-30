# Categorize YOLO/COCO classes into semantic groups

# COCO classes relevant to our reasoning
# This is a simple dictionary mapping broad categories to specific object names.

CATEGORIES = {
    'liquid': ['cup', 'bottle', 'wine glass', 'bowl'],
    'electronics': ['laptop', 'mouse', 'keyboard', 'cell phone', 'tv', 'remote'],
    'flammable': ['book', 'paper', 'cardboard box'],
    'sharp': ['knife', 'scissors', 'fork'],
    'furniture': ['dining table', 'chair', 'couch', 'bed']
}

def get_categories(class_name):
    """
    Returns a list of categories for a given object class name.
    Example: 'cup' -> ['liquid']
    """
    found_categories = []
    
    for category, items in CATEGORIES.items():
        if class_name in items:
            found_categories.append(category)
            
    return found_categories

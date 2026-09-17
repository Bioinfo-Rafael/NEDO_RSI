# ORIGIN: ORIGINAL
# IMPLEMENTATION_NOTE: The only editable policy program during autoresearch.

def choose(ctx):
    """Rank observed opportunities. This file uses the documented safe Python subset."""
    if ctx['budget_remaining'] <= 0 or not ctx['frontier']:
        return {'action': 'STOP', 'target_node_id': None, 'reason': 'finished'}
    best = None
    best_value = -1000000
    for node in ctx['frontier']:
        value = node['observed_rank']
        if value > best_value:
            best = node
            best_value = value
    action = 'DEEPEN'
    if best['children_count'] > 0:
        action = 'WIDEN'
    elif best['node_id'] != ctx['current_node_id']:
        action = 'REVISIT'
    return {'action': action, 'target_node_id': best['node_id'], 'reason': 'prioritize observed quality without depth bonus'}

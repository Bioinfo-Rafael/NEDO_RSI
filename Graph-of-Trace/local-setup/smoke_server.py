"""TEST ONLY: deterministic extraction; normal got-server.sh never loads this."""
from Monitor import steps_llm


async def fake_build_nodes(*, session_id, subtask, artifacts, steps):
    assert session_id == 'smoke-test'
    assert subtask['title'] == 'Verified Graph of Trace installation'
    return [{
        'id': 'N002', 'title': subtask['title'],
        'description': subtask['description'],
        'parents': [{'id': 'N001', 'relation': 'necessitated_by'}],
        'artifacts': artifacts,
    }]


steps_llm.build_nodes = fake_build_nodes
from server import main

if __name__ == '__main__':
    main()

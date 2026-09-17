from .util import stable_id


def add(con, entity_type, entity_id, source_id, file_id=None, line_no=None, event_index=None,
        git_commit=None, json_pointer=None, derivation_method="direct", confidence=1.0, evidence=None):
    pid=stable_id("prov",entity_type,entity_id,source_id,file_id,line_no,event_index,git_commit,json_pointer,derivation_method)
    con.execute("INSERT OR IGNORE INTO provenance VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (pid,entity_type,entity_id,source_id,file_id,line_no,event_index,git_commit,json_pointer,derivation_method,confidence,evidence))
    return pid

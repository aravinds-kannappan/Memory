"""Typed information graph, purpose masks, provenance withdrawal, exact path tests."""
import heapq
import networkx as nx
import numpy as np

NODE_TYPES={'video':'observation','skeleton':'movement_features','ehr':'ehr_event',
 'gate':'model_output','summary':'summary','prediction':'model_output','pager':'review',
 'physician':'review','action':'clinical_action','ehr_write':'persistent_storage',
 'archive':'persistent_storage','profile':'persistent_storage','cache':'summary'}

def pipeline(bypass=False, cache=True):
    g=nx.DiGraph()
    for n,typ in NODE_TYPES.items():
        lineage=['ehr'] if n=='ehr' else ['video'] if n in ['video','skeleton','archive'] else ['video','ehr']
        g.add_node(n,type=typ,timestamp=0.,patient='p0',provenance=lineage)
    edges=[('video','skeleton'),('skeleton','gate'),('ehr','gate'),('gate','summary'),
           ('summary','prediction'),('prediction','pager'),('pager','physician'),
           ('physician','action'),('physician','ehr_write'),('video','archive'),('summary','profile')]
    if cache:edges += [('summary','cache'),('cache','prediction'),('cache','archive')]
    if bypass:edges += [('summary','action')]
    for u,v in edges:
        typ='reviewed by' if v in ['pager','physician'] else 'used to trigger' if v in ['action','ehr_write'] else 'queried by' if u=='ehr' else 'derived from'
        purposes=['fall_detection','archiving','profiling']
        g.add_edge(u,v,type=typ,timestamp=0.,latency=1.,purposes=purposes,patient='p0')
    return g

def mask(g, patient, purpose, permissions, time=0, provenance=True):
    """Permissions map (patient, source, purpose) to bool or withdrawal timestamp.

    Missing entries deny. Sink scopes prevent purpose laundering; provenance
    checks also disable stored derivatives after source withdrawal.
    """
    def allowed(source):
        value=permissions.get((patient,source,purpose),False)
        return value if isinstance(value,bool) else time<float(value)
    nodes=list(g); A=nx.to_numpy_array(g,nodelist=nodes,weight=None); M=np.zeros_like(A)
    index={n:i for i,n in enumerate(nodes)}
    for u,v,e in g.edges(data=True):
        ok=(e.get('patient',patient)==patient and g.nodes[u].get('patient',patient)==patient and
            g.nodes[v].get('patient',patient)==patient and purpose in e['purposes'])
        sources=g.nodes[u].get('provenance',[u]) if provenance else ([u] if u in ['video','ehr'] else [])
        ok=ok and all(allowed(s) for s in sources)
        # Any persistent derivative counts as storage, including EHR writes.
        if g.nodes[v]['type']=='persistent_storage':
            ok=ok and purpose in ('archiving','profiling')
        if ok:M[index[u],index[v]]=1
    out=g.copy();out.remove_edges_from([(u,v) for u,v in g.edges if not M[index[u],index[v]]])
    assert np.array_equal(nx.to_numpy_array(out,nodelist=nodes,weight=None), A*M)
    return out,A,M

def violating_paths(g,sources,decisions,reviews):
    if set(sources)&set(reviews) or set(decisions)&set(reviews):raise ValueError('Review nodes must not be endpoints')
    reduced=g.copy();reduced.remove_nodes_from(reviews);paths=[]
    for s in sources:
        for d in decisions:
            if s in reduced and d in reduced:
                paths.extend(list(nx.all_simple_paths(reduced,s,d)))
    return paths

def temporal_path(g,source,target,start=0,deadline=float('inf')):
    """Earliest-arrival path with nondecreasing edge event times and latency."""
    todo=[(start,source,[source])];best={source:start}
    while todo:
        t,u,path=heapq.heappop(todo)
        if u==target:return path,t
        if t>best[u]:continue
        for v,e in g[u].items():
            event=e.get('timestamp',t)
            if e.get('event_only',False) and event<t:continue
            arrive=max(t,event)+e.get('latency',0)
            if arrive<=deadline and arrive<best.get(v,float('inf')):
                best[v]=arrive;heapq.heappush(todo,(arrive,v,path+[v]))
    return None,None

def protected_vertex_cut(g,source,target):
    if not nx.has_path(g,source,target):return []
    if g.has_edge(source,target):return None  # no internal vertex cut exists
    return sorted(nx.minimum_node_cut(g,source,target))

def revoke_artifacts(artifacts,patient,source,purpose,time):
    """Invalidate future use, including derivatives; no claim of physical erasure."""
    changed=[]
    for a in artifacts:
        if a['patient']==patient and source in a['provenance']:
            a.setdefault('revoked',{})[purpose]=time;changed.append(a['id'])
    return changed

def usable(artifact,purpose,time):
    return time<artifact.get('revoked',{}).get(purpose,float('inf'))

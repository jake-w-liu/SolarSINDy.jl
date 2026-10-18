"""Data-only, bounded BSON/Julia-tag reader. Never raises or resolves Julia types."""
from dataclasses import dataclass
from pathlib import Path
import bisect
import math
import struct


class InvalidArtifact(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise InvalidArtifact(message)


@dataclass(frozen=True)
class TypeSpec:
    name: tuple
    params: tuple


@dataclass(frozen=True)
class NumericArray:
    dtype: str
    shape: tuple
    values: tuple


@dataclass(frozen=True)
class PlainStruct:
    type: TypeSpec
    fields: tuple


class Reader:
    def __init__(self, data, *, max_bytes=64*1024*1024, max_depth=64,
                 max_elements=2_000_000, max_string=1024*1024):
        require(isinstance(data, bytes) and len(data) <= max_bytes, 'BSON byte budget')
        self.data, self.pos = data, 0
        self.max_depth, self.max_elements = max_depth, max_elements
        self.max_string, self.elements = max_string, 0

    def take(self, n, end):
        require(isinstance(n, int) and 0 <= n <= end-self.pos, 'truncated BSON payload')
        value = self.data[self.pos:self.pos+n]
        self.pos += n
        return value

    def cstring(self, end):
        stop = self.data.find(b'\0', self.pos, min(end, self.pos+self.max_string+1))
        require(stop >= self.pos, 'unterminated/oversized BSON key')
        value = self.take(stop-self.pos, end).decode('utf-8', 'strict')
        self.take(1, end)
        return value

    def document(self, end, depth=0, array=False):
        require(depth <= self.max_depth, 'BSON nesting budget')
        start = self.pos
        length = struct.unpack('<i', self.take(4, end))[0]
        require(5 <= length <= end-start, 'invalid BSON document length')
        limit = start+length
        result = {}
        while self.pos < limit-1:
            self.elements += 1
            require(self.elements <= self.max_elements, 'BSON element budget')
            tag = self.take(1, limit-1)[0]
            name = self.cstring(limit-1)
            require(name not in result, 'duplicate BSON document key')
            if tag == 1:
                value = struct.unpack('<d', self.take(8, limit-1))[0]
            elif tag == 2:
                count = struct.unpack('<i', self.take(4, limit-1))[0]
                require(1 <= count <= self.max_string, 'BSON string budget/length')
                raw = self.take(count, limit-1)
                require(raw[-1:] == b'\0', 'BSON string lacks terminator')
                value = raw[:-1].decode('utf-8', 'strict')
            elif tag in (3, 4):
                value = self.document(limit-1, depth+1, array=(tag == 4))
            elif tag == 5:
                count = struct.unpack('<i', self.take(4, limit-1))[0]
                subtype = self.take(1, limit-1)[0]
                require(subtype == 0, 'unsupported BSON binary subtype')
                value = self.take(count, limit-1)
            elif tag == 8:
                flag = self.take(1, limit-1)[0]
                require(flag in (0, 1), 'invalid BSON bool')
                value = bool(flag)
            elif tag == 10:
                value = None
            elif tag in (16, 18):
                width, fmt = (4, '<i') if tag == 16 else (8, '<q')
                value = struct.unpack(fmt, self.take(width, limit-1))[0]
            else:
                raise InvalidArtifact('unsupported BSON primitive tag '+str(tag))
            result[name] = value
        require(self.pos == limit-1 and self.take(1, limit) == b'\0', 'invalid BSON end marker')
        if array:
            require(list(result) == [str(i) for i in range(len(result))], 'non-sequential BSON array keys')
            return list(result.values())
        return result

    def parse(self):
        result = self.document(len(self.data))
        require(self.pos == len(self.data), 'trailing BSON bytes')
        return result


def parse_bytes(data, **limits):
    return Reader(data, **limits).parse()


def read_bson(path, *, max_bytes=64*1024*1024):
    require(type(max_bytes) is int and max_bytes>0, 'positive integer BSON byte budget required')
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'BSON regular file required')
    with path.open('rb') as stream:
        data = stream.read(max_bytes+1)
    require(len(data) <= max_bytes, 'BSON byte budget')
    return parse_bytes(data, max_bytes=max_bytes)


NUMERIC = {'Int64': ('q', 8), 'Int32': ('i', 4), 'UInt8': ('B', 1),
           'Float32': ('f', 4), 'Float64': ('d', 8), 'Bool': ('B', 1)}
ALLOWED_TYPES = {('Core', name) for name in (*NUMERIC, 'Any', 'Symbol', 'String')}
ALLOWED_TYPES |= {('EvoTrees', 'EvoTree'), ('EvoTrees', 'Tree'),
                  ('EvoTrees', 'MSE')}


class Decoder:
    def __init__(self, root, *, max_values=2_000_000, max_depth=96):
        require(isinstance(root, dict), 'BSON root document required')
        self.root = root
        self.refs = root.get('_backrefs', [])
        require(isinstance(self.refs, list) and len(self.refs) <= max_values, 'invalid backreference table')
        self.max_values, self.max_depth, self.values = max_values, max_depth, 0
        self.cache, self.active = {}, set()
        self.references_validated = False
        self.heights = {}

    def bounded(self, value):
        """Limit expanded output nesting even when a child came from the reference cache.

        The root/container counts one level; scalar values count zero. Strong memo
        entries prevent identity reuse while avoiding repeated walks through shared DAGs.
        """
        def height(item):
            if isinstance(item, (str,int,float,bool)) or item is None:return 0
            cached=self.heights.get(id(item))
            if cached is not None and cached[0] is item:return cached[1]
            if isinstance(item, (NumericArray,bytes)):result=1
            else:
                if isinstance(item,dict):children=item.values()
                elif isinstance(item,tuple):children=item
                elif isinstance(item,TypeSpec):children=item.params
                elif isinstance(item,PlainStruct):children=(*item.fields,item.type)
                else:raise InvalidArtifact('unexpected decoded object type')
                result=1+max((height(child) for child in children),default=0)
            self.heights[id(item)]=(item,result)
            return result
        require(height(value)<=self.max_depth,'expanded decoded nesting budget')
        return value

    def reference(self, index, depth):
        require(type(index) is int and 1 <= index <= len(self.refs), 'backreference index')
        require(index not in self.active, 'cyclic backreference')
        if index not in self.cache:
            self.active.add(index)
            self.cache[index] = self.decode(self.refs[index-1], depth+1)
            self.active.remove(index)
        return self.bounded(self.cache[index])

    def decode(self, value, depth=0):
        if depth == 0 and not self.references_validated:
            # Even unreferenced table entries must satisfy the same data-only schema.
            for index in range(1,len(self.refs)+1):self.reference(index,1)
            self.references_validated = True
        self.values += 1
        require(self.values <= self.max_values and depth <= self.max_depth, 'decoded data budget')
        if isinstance(value, list):
            return self.bounded(tuple(self.decode(x, depth+1) for x in value))
        if not isinstance(value, dict):
            if isinstance(value, bytes):
                self.values += len(value)
                require(self.values <= self.max_values, 'decoded numeric/binary element budget')
            return self.bounded(value)
        tag = value.get('tag')
        if tag is None:
            return self.bounded({k: self.decode(v, depth+1) for k, v in value.items() if k != '_backrefs'})
        if tag == 'backref':
            require(set(value) == {'tag', 'ref'}, 'backreference schema')
            return self.reference(value['ref'],depth)
        if tag == 'symbol':
            require(set(value) == {'tag', 'name'} and isinstance(value['name'], str), 'symbol schema')
            return value['name']
        if tag == 'datatype':
            require(set(value) == {'tag', 'name', 'params'}, 'datatype schema')
            require(isinstance(value['name'], list), 'datatype name schema')
            name = tuple(value['name'])
            require(name in ALLOWED_TYPES, 'unrecognized data-only Julia type '+str(name))
            require(isinstance(value['params'],list),'datatype parameters schema')
            if name in {('Core',name) for name in NUMERIC}:
                require(not value['params'],'primitive numeric type cannot have parameters')
            elif name[0]=='Core' or name==('EvoTrees','MSE'):
                require(not value['params'],'nonparametric datatype cannot have parameters')
            params=tuple(self.decode(x, depth+1) for x in value['params'])
            if name in (('EvoTrees','Tree'),('EvoTrees','EvoTree')):
                require(len(params)==2 and params[0]==TypeSpec(('EvoTrees','MSE'),()) and
                        type(params[1]) is int and params[1]==1,
                        'unsupported scalar MSE type parameters')
            return self.bounded(TypeSpec(name,params))
        if tag in ('tuple', 'svec'):
            require(set(value) == {'tag', 'data'} and isinstance(value['data'], list), 'tuple schema')
            return self.bounded(tuple(self.decode(x, depth+1) for x in value['data']))
        if tag == 'array':
            require(set(value) == {'tag', 'type', 'size', 'data'}, 'numeric array schema')
            dtype = self.decode(value['type'], depth+1)
            require(isinstance(dtype, TypeSpec) and dtype.name[:1] == ('Core',) and not dtype.params,
                    'numeric array type schema')
            name = dtype.name[-1]
            require(name in NUMERIC and isinstance(value['size'], list), 'unsupported numeric array')
            shape = tuple(value['size'])
            require(1 <= len(shape) <= 2 and all(type(n) is int and 0 <= n <= self.max_values for n in shape),
                    'numeric array dimensions')
            count = math.prod(shape)
            require(count <= self.max_values, 'numeric array element budget')
            raw = value['data']
            fmt, width = NUMERIC[name]
            require(isinstance(raw, bytes) and len(raw) == count*width, 'numeric array byte count')
            self.values += count
            require(self.values <= self.max_values, 'decoded numeric/binary element budget')
            vals = tuple(x[0] for x in struct.iter_unpack('<'+fmt, raw))
            if name == 'Bool':
                require(all(x in (0, 1) for x in vals), 'numeric Bool encoding')
                vals = tuple(bool(x) for x in vals)
            return self.bounded(NumericArray(name, shape, vals))
        if tag == 'struct':
            require(set(value) == {'tag', 'type', 'data'}, 'struct schema')
            dtype = self.decode(value['type'], depth+1)
            require(isinstance(dtype, TypeSpec), 'struct type schema')
            if dtype.name in {('Core', name) for name in NUMERIC}:
                require(not dtype.params, 'primitive numeric type cannot have parameters')
                fmt, width = NUMERIC[dtype.name[-1]]
                require(isinstance(value['data'], bytes) and len(value['data']) == width, 'primitive struct width')
                number = struct.unpack('<'+fmt, value['data'])[0]
                if dtype.name[-1] == 'Bool':
                    require(number in (0, 1), 'primitive Bool encoding')
                    number = bool(number)
                return number
            require(isinstance(value['data'], list), 'struct field list')
            fields = tuple(self.decode(x, depth+1) for x in value['data'])
            require(dtype.name in (('EvoTrees', 'EvoTree'), ('EvoTrees', 'Tree')), 'unexpected struct')
            return self.bounded(PlainStruct(dtype, fields))
        raise InvalidArtifact('unsupported data tag '+str(tag))


def decode_root(root):
    return Decoder(root).decode(root)


def f32(value):
    result = struct.unpack('<f', struct.pack('<f', value))[0]
    require(math.isfinite(result), 'Float32 prediction overflow')
    return result


def vector(value, dtype, count):
    if dtype == 'UInt8' and isinstance(value, bytes):
        require(len(value) == count, 'wrong raw UInt8 vector shape')
        return tuple(value)
    require(isinstance(value, NumericArray) and value.dtype == dtype and value.shape == (count,),
            'wrong '+dtype+' vector shape')
    return value.values


def validate_model(document, names, configuration, fit_count=None):
    require(set(document) == {'model'}, 'EvoTrees BSON root schema')
    model = document['model']
    mse = TypeSpec(('EvoTrees', 'MSE'), ())
    require(isinstance(model, PlainStruct) and model.type.name == ('EvoTrees','EvoTree')
            and len(model.type.params)==2 and model.type.params[0]==mse
            and type(model.type.params[1]) is int and model.type.params[1]==1,
            'model must be scalar EvoTree MSE')
    require(len(model.fields) == 4, 'EvoTree field count')
    loss, k, trees, info = model.fields
    require(loss == mse and type(k) is int and k == 1, 'model loss/K mismatch')
    require(isinstance(trees, tuple) and len(trees) == configuration['nrounds']+1, 'wrong GBM tree count')
    require(isinstance(info, dict), 'model info schema')
    require(set(info) == {'nrounds','feature_names','target_levels','target_isordered','edges','featbins','feattypes','logger'},
            'model info key set')
    require(info['nrounds'] == configuration['nrounds'], 'GBM round metadata')
    require(tuple(info['feature_names']) == tuple(names) and len(names) == 29 and len(set(names)) == 29,
            'GBM ordered feature schema')
    require(info['target_levels'] is None and info['target_isordered'] is False, 'GBM target type metadata')
    require(info['logger'] is None, 'unexpected GBM training logger schema')
    edges = info['edges']
    require(isinstance(edges, tuple) and len(edges) == 29, 'GBM edge dimensions')
    featbins = vector(info['featbins'], 'UInt8', 29)
    feattypes = vector(info['feattypes'], 'Bool', 29)
    edge_values = []
    for j, values in enumerate(edges):
        require(isinstance(values, NumericArray) and values.dtype == 'Float64'
                and values.shape == (configuration['nbins']-1,), 'GBM edge schema/bin count')
        vals = values.values
        require(all(math.isfinite(x) for x in vals) and all(a <= b for a,b in zip(vals,vals[1:])), 'GBM edge order/finite')
        require(featbins[j] == len(vals)+1 and feattypes[j] is True, 'GBM numeric bin metadata')
        edge_values.append(vals)
    checked = []
    for index, tree in enumerate(trees):
        require(isinstance(tree, PlainStruct) and tree.type.name==('EvoTrees','Tree') and
                len(tree.type.params)==2 and tree.type.params[0]==mse and
                type(tree.type.params[1]) is int and tree.type.params[1]==1, 'Tree type')
        require(len(tree.fields) == 6, 'Tree field count')
        feat, cond, weights, gains, pred, split = tree.fields
        nodes = 1 if index == 0 else 2**configuration['max_depth']-1
        feat = vector(feat, 'Int64', nodes)
        cond = vector(cond, 'UInt8', nodes)
        weights = vector(weights, 'Float32', nodes)
        gains = vector(gains, 'Float32', nodes)
        split = vector(split, 'Bool', nodes)
        require(isinstance(pred, NumericArray) and pred.dtype == 'Float32' and pred.shape == (1,nodes), 'Tree predictions shape')
        require(all(math.isfinite(x) for x in (*weights,*gains,*pred.values)), 'nonfinite Tree values')
        require(all(x >= 0 for x in weights), 'negative Tree sample weights')
        require(all(0 <= x <= 29 for x in feat) and all(0 <= x < configuration['nbins'] for x in cond),
                'Tree stored feature/bin range')
        if index and fit_count is not None:
            require(type(fit_count) is int and 2 <= fit_count <= 1_000_000 and weights[0] == f32(fit_count),
                    'Tree root sample count differs from exact full fit ledger')
        todo = [(0,1)]
        reachable = set()
        while todo:
            node, depth = todo.pop()
            reachable.add(node)
            require(depth <= configuration['max_depth'] or index == 0, 'Tree maximum depth')
            if split[node]:
                require(index != 0 and 1 <= feat[node] <= 29, 'Tree split feature')
                require(1 <= cond[node] < featbins[feat[node]-1], 'Tree split bin')
                require(2*node+2 < nodes, 'Tree child outside stored heap')
                require(weights[2*node+1]+weights[2*node+2] == weights[node], 'Tree child sample counts')
                todo.extend(((2*node+1,depth+1),(2*node+2,depth+1)))
        require(all(not split[node] for node in range(nodes) if node not in reachable), 'unreachable split in Tree heap')
        if index == 0:
            require(split == (False,), 'bias tree cannot split')
        checked.append((feat,cond,pred.values,split))
    return tuple(edge_values), tuple(checked), info


def predict(edges, trees, values):
    require(len(values) == 29 and all(math.isfinite(x) for x in values), 'finite 29 GBM values required')
    bins = tuple(bisect.bisect_left(edge,x)+1 for edge,x in zip(edges,values))
    result = 0.0
    for feat,cond,pred,split in trees:
        node = 0
        while split[node]:
            node = 2*node+1 if bins[feat[node]-1] <= cond[node] else 2*node+2
        result = f32(result+pred[node])
    return result

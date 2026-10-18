"""Manual literal/tag/bin/Float32 expectations frozen before implementation."""
from pathlib import Path
import bisect
import json
import struct
import unittest
import sys
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/initial_reference_audit'))
from bounded_bson import (parse_bytes, Decoder, InvalidArtifact, NumericArray,
                          f32, predict)

HERE = Path(__file__).resolve().parent/'frozen'
assert hashlib.sha256((HERE/'FROZEN_ORACLES.json').read_bytes()).hexdigest()=='dd9ef99c6d14757e19d9aa3e898ca24601b231941c19b6070188bac1f53349ef'
ORACLE = json.loads((HERE/'FROZEN_ORACLES.json').read_text(encoding='utf-8'))


def typed(name, shape, raw):
    return {'tag':'array','type':{'tag':'datatype','name':['Core',name],'params':[]},
            'size':shape,'data':raw}


class ManualOracles(unittest.TestCase):
    def test_frozen_literals(self):
        for case in ORACLE['bson_literals']:
            with self.subTest(case=case['name']):
                if case.get('reject'):
                    with self.assertRaises(InvalidArtifact):parse_bytes(bytes.fromhex(case['hex']))
                else:self.assertEqual(parse_bytes(bytes.fromhex(case['hex'])),case['expected'])

    def test_frozen_arrays(self):
        for case in ORACLE['typed_arrays']:
            node=typed(case['type'][-1],case['size'],bytes.fromhex(case['data_hex']))
            with self.subTest(case=case):
                if case.get('reject'):
                    with self.assertRaises(InvalidArtifact):Decoder({}).decode(node)
                else:self.assertEqual(Decoder({}).decode(node).values,tuple(case['expected']))

    def test_backrefs(self):
        item=typed('Float32',[1],struct.pack('<f',0.5))
        ref={'tag':'backref','ref':1}
        root={'_backrefs':[item],'a':ref,'b':ref}
        got=Decoder(root).decode(root)
        self.assertEqual(got['a'].values,(0.5,))
        self.assertIs(got['a'],got['b'])
        for root in ({'_backrefs':[item],'a':{'tag':'backref','ref':0}},
                     {'_backrefs':[item],'a':{'tag':'backref','ref':2}},
                     {'_backrefs':[ref],'a':ref},
                     {'a':{'tag':'ref','path':['Main','danger']}}):
            with self.assertRaises(InvalidArtifact):Decoder(root).decode(root)

    def test_unused_refs_and_primitive_parameters(self):
        for item in ({'tag':'ref','path':['Main','danger']},{'tag':'backref','ref':1}):
            root={'_backrefs':[item],'ordinary':1}
            with self.assertRaises(InvalidArtifact):Decoder(root).decode(root)
        primitive={'tag':'struct','type':{'tag':'datatype','name':['Core','UInt8'],
                   'params':[{'tag':'datatype','name':['Core','Any'],'params':[]}]},'data':b'\1'}
        with self.assertRaises(InvalidArtifact):Decoder({}).decode(primitive)

    def test_aggregate_numeric_budget(self):
        first=typed('Float64',[8],struct.pack('<8d',*range(8)))
        self.assertEqual(len(Decoder({},max_values=10).decode(first).values),8)
        with self.assertRaises(InvalidArtifact):Decoder({},max_values=10).decode([first,first])
        with self.assertRaises(InvalidArtifact):Decoder({},max_values=10).decode(list(range(16)))

    def test_cached_expansion_exact_depth(self):
        def chain(levels):
            refs=[7]
            for i in range(1,levels+1):refs.append([{'tag':'backref','ref':i}])
            return {'_backrefs':refs,'x':{'tag':'backref','ref':len(refs)}}
        root=chain(5)
        value=Decoder(root,max_depth=6).decode(root)['x']
        for _ in range(5):value=value[0]
        self.assertEqual(value,7)
        for levels in (6,15):
            root=chain(levels)
            with self.assertRaises(InvalidArtifact):Decoder(root,max_depth=6).decode(root)
        impossible={'tag':'datatype','name':['Core','UInt8'],'params':[{'tag':'datatype','name':['Core','Any'],'params':[]}]}
        root={'_backrefs':[impossible],'value':7}
        with self.assertRaises(InvalidArtifact):Decoder(root).decode(root)

    def test_complete_type_grammar(self):
        def dtype(module,name,params=None):return {'tag':'datatype','name':[module,name],'params':[] if params is None else params}
        mse=dtype('EvoTrees','MSE');scalar=dtype('Core','Float32')
        for module,name in [('Core',name) for name in ('Int64','Int32','UInt8','Float32','Float64','Bool','Any','Symbol','String')]+[('EvoTrees','MSE')]:
            self.assertEqual(Decoder({}).decode(dtype(module,name)).params,())
            for params in ([scalar],[1],[mse,1]):
                with self.assertRaises(InvalidArtifact):Decoder({}).decode(dtype(module,name,params))
        for name in ('Tree','EvoTree'):
            self.assertEqual(Decoder({}).decode(dtype('EvoTrees',name,[mse,1])).params[1],1)
            for params in ([],[mse],[mse,1,scalar],[scalar,1],[mse,0],[mse,2],[mse,True],[mse,1.0]):
                for unused in (False,True):
                    node=dtype('EvoTrees',name,params)
                    root={'_backrefs':[node],'ordinary':7} if unused else {'x':node}
                    with self.assertRaises(InvalidArtifact):Decoder(root).decode(root)
        for params in ([],[dtype('Core','Symbol')],[dtype('Core','Symbol'),dtype('Core','Any')]):
            with self.assertRaises(InvalidArtifact):Decoder({}).decode(dtype('Base','Dict',params))
        self.assertEqual(Decoder({}).decode({'info':{'text':'metadata','flag':True}}),{'info':{'text':'metadata','flag':True}})

    def test_budgets(self):
        raw=bytes.fromhex(ORACLE['bson_literals'][0]['hex'])
        with self.assertRaises(InvalidArtifact):parse_bytes(raw,max_bytes=8)
        with self.assertRaises(InvalidArtifact):parse_bytes(raw,max_elements=0)
        with self.assertRaises(InvalidArtifact):Decoder({},max_values=1).decode([1,2])
        with self.assertRaises(InvalidArtifact):Decoder({}).decode({'tag':'datatype','name':['Main','Danger'],'params':[]})

    def test_resource_and_array_keys(self):
        # Nested array {x:[42]} encoded independently of the implementation.
        raw=bytes.fromhex('140000000478000c0000001030002a0000000000')
        self.assertEqual(parse_bytes(raw),{'x':[42]})
        with self.assertRaises(InvalidArtifact):parse_bytes(raw,max_depth=0)
        with self.assertRaises(InvalidArtifact):parse_bytes(raw.replace(b'\x10' + b'0\0',b'\x10' + b'1\0'))
        # {b: binary subtype1, empty} is outside this format.
        with self.assertRaises(InvalidArtifact):parse_bytes(bytes.fromhex('0d000000056200000000000100'))

    def test_bins_and_heap(self):
        case=ORACLE['bin_edges']
        self.assertEqual([bisect.bisect_left(case['edges'],x)+1 for x in case['values']],case['expected_one_based_bins'])
        edges=(tuple(case['edges']),)*29
        # Independent hand encoding of one bias, one root split and a constant.
        trees=(((0,),(0,),(0.5,),(False,)),
               ((1,0,0),(2,0,0),(0.,2.,-3.),(True,False,False)),
               ((0,),(0,),(0.25,),(False,)))
        for x,expected in zip((-2.,0.,1.,2.,3.),ORACLE['tree']['expected_delta']):
            self.assertEqual(predict(edges,trees,(x,)*29),expected)
        total=0.
        for term in ORACLE['tree']['float32_order_control']['terms']:total=f32(total+term)
        self.assertEqual(total,0.)


if __name__=='__main__':unittest.main(verbosity=2)

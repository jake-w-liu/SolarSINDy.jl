from pathlib import Path
from datetime import datetime
from decimal import Decimal,localcontext
import importlib.util,math,struct,unittest
P=Path(__file__).resolve().parents[2]/'tools/point_input_export/input_values.py'
spec=importlib.util.spec_from_file_location('point_input_verifier_under_test',P)
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)

def fixture(width):
    rows=[]
    for k in range(121):
        feature=bytearray(160);label=bytearray(64)
        struct.pack_into('<qqBB',feature,0,k*60000,k*60000+1845000,int(k not in (37,52)),0 if k not in (37,52) else 1)
        struct.pack_into('<qqBB',label,0,k*60000,k*60000+1845000,int(k!=41),0 if k!=41 else 3)
        struct.pack_into('<dd',feature,24,float(k),1.+k/10)
        for j in range(width):struct.pack_into('<d',feature,40+8*j,(-1. if j==7 else 1.)*(100*k+j+1))
        struct.pack_into('<d',label,24,5. if k%2 else 10.)
        rows.append((k+1,bytes(feature),bytes(label)))
    # Literal independently derived selection:35..59 inclusive, less three
    # unaccepted rows; target at minute60 becomes ready after the split stop.
    selected=[k for k in range(35,60) if k not in (37,41,52)]
    n=len(selected);chunks=[b'SOURCE_BOUND_POINT_INPUT_V1\n',struct.pack('<QQ',n,width)]
    chunks.append(struct.pack('<'+'Q'*n,*[k+1 for k in selected]))
    rata_die=datetime(1970,1,1).toordinal()*86400000
    chunks.append(struct.pack('<'+'q'*n,*[rata_die+k*60000 for k in selected]))
    for j in range(width):chunks.append(b''.join(rows[k][1][40+8*j:48+8*j] for k in selected))
    chunks.append(b''.join(rows[k][2][24:32] for k in selected))
    with localcontext() as context:
        context.prec=100
        logs=[float(Decimal(6 if k%2 else 11).ln()) for k in selected]
    chunks.append(struct.pack('<'+'d'*n,*logs))
    chunks.append(b''.join(rows[k][1][24:32] for k in selected))
    chunks.append(b''.join(rows[k][1][32:40] for k in selected))
    return b''.join(chunks),rows,selected

class InputOracleTests(unittest.TestCase):
    def check(self,data,records,width=8,**extra):
        return V.check_input_bytes(data,records,start_ms=35*60000,stop_ms=90*60000,asof_ms=120*60000,width=width,**extra)
    def test_both_layouts_literal_rows_and_decimal_logs(self):
        for width in (8,11):
            data,rows,wanted=fixture(width)
            result=self.check(data,rows,width)
            self.assertEqual(result['rows'],22);self.assertEqual(result['columns'],width)
            self.assertEqual((result['first_row'],result['last_row']),(36,60))
            self.assertEqual(result['scheduled_rows_scanned'],121)
            self.assertTrue(result['all_source_input_coordinates_byte_exact'])
            self.assertEqual(result['transformed_targets_independently_checked'],22)
            self.assertEqual(result['actual_model_fits'],0)
    def test_semantic_corruptions(self):
        data,rows,wanted=fixture(8);base=len(V.PREFIX)+16;n=22
        offsets=dict(row=base,time=base+8*n,feature=base+16*n,target=base+8*n*(2+8),
                     log=base+8*n*(3+8),raw=base+8*n*(4+8),scale=base+8*n*(5+8))
        variants=[data+b'\0',data[:-1],b'X'+data[1:]]
        for label,offset in offsets.items():
            value=bytearray(data)
            if label=='row':struct.pack_into('<Q',value,offset,37)
            elif label=='time':struct.pack_into('<q',value,offset,35*60000)
            else:struct.pack_into('<d',value,offset,123456.)
            variants.append(bytes(value))
        for value in variants:
            with self.assertRaises(ValueError):self.check(value,rows)
        for nbad,pbad in ((1,8),(23,8),(22,9),(600001,8)):
            value=bytearray(data);struct.pack_into('<QQ',value,len(V.PREFIX),nbad,pbad)
            with self.assertRaises(ValueError):self.check(bytes(value),rows)
    def test_split_boundary_equality_and_no_future_target_read(self):
        data,rows,_=fixture(8)
        altered=list(rows)
        row,feature,label=altered[75];value=bytearray(label);struct.pack_into('<d',value,24,float('nan'))
        altered[75]=(row,feature,bytes(value))
        self.assertEqual(self.check(data,altered)['input_sha256'],self.check(data,rows)['input_sha256'])
        with self.assertRaises(ValueError):
            V.check_input_bytes(data,rows,start_ms=35*60000,stop_ms=59*60000+1845000,asof_ms=120*60000,width=8)
        with self.assertRaises(ValueError):self.check(data,rows[1:])
        row,feature,label=rows[1];bad=list(rows);bad[1]=(True,feature,label)
        with self.assertRaises(ValueError):self.check(data,bad)
    def test_duplicate_permutation_and_nonfinite_native_transform(self):
        data,rows,_=fixture(11);base=len(V.PREFIX)+16;n=22
        bad=list(rows);bad[15]=bad[14]
        with self.assertRaises(ValueError):self.check(data,bad,11)
        bad=list(rows);bad[13],bad[14]=bad[14],bad[13]
        with self.assertRaises(ValueError):self.check(data,bad,11)
        changed=bytearray(data);struct.pack_into('<d',changed,base+8*n*(3+11),float('nan'))
        with self.assertRaises(ValueError):self.check(bytes(changed),rows,11)
    def test_bad_indices_and_clock_geometry(self):
        data,rows,_=fixture(8)
        for width in (True,False,9,8.):
            with self.assertRaises(ValueError):self.check(data,rows,width)
        with self.assertRaises(ValueError):V.check_input_bytes(data,rows,start_ms=0,stop_ms=0,asof_ms=0,width=8)
        with self.assertRaises(ValueError):V.check_input_bytes(bytearray(data),rows,start_ms=0,stop_ms=1,asof_ms=2,width=8)

if __name__=='__main__':unittest.main(verbosity=2)
